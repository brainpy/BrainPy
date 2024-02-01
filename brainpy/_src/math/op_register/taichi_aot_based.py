import contextlib
import hashlib
import inspect
import io
import os
import pathlib
import platform
import re
import shutil
from functools import partial, reduce
from typing import Any, Sequence

import jax.core
import numpy as np
from jax.interpreters import xla
from jax.lib import xla_client

from brainpy._src.dependency_check import import_taichi, import_brainpylib_cpu_ops, import_brainpylib_gpu_ops
from .utils import _shape_to_layout


### UTILS ###

# get the path of home directory on Linux, Windows, Mac
def get_home_dir():
  return str(pathlib.Path.home())


# encode a string with md5
def encode_md5(source: str) -> str:
  # create md5 object
  md5 = hashlib.md5()

  # encode source
  source_encode = source.encode(encoding='utf-8')

  # update md5 object
  md5.update(source_encode)

  return md5.hexdigest()

# check kernels count
def check_kernels_count() -> int:
  if not os.path.exists(kernels_aot_path):
    return 0
  kernels_count = 0
  dir1 = os.listdir(kernels_aot_path)
  for i in dir1:
    dir2 = os.listdir(os.path.join(kernels_aot_path, i))
    kernels_count += len(dir2)
  return kernels_count

# clean caches
def clean_caches(kernels_name: list[str]=None):
  if kernels_name is None:
    if not os.path.exists(kernels_aot_path):
      raise FileNotFoundError("The kernels cache folder does not exist. \
                              Please define a kernel using `taichi.kernel` \
                              and customize the operator using `bm.XLACustomOp` \
                              before calling the operator.")
    shutil.rmtree(kernels_aot_path)
    print('Clean all kernel\'s cache successfully')
    return
  for kernel_name in kernels_name:
    try:
      shutil.rmtree(os.path.join(kernels_aot_path, kernel_name))
    except FileNotFoundError:
      raise FileNotFoundError(f'Kernel {kernel_name} does not exist.')
  print('Clean kernel\'s cache successfully')

# TODO
# not a very good way
# get source with dependencies
def get_source_with_dependencies(func, visited=None):
  if visited is None:
    visited = set()

  source = inspect.getsource(func)
  if func in visited:
    return ''

  visited.add(func)
  module = inspect.getmodule(func)
  dependent_funcs = re.findall(r'(\w+)\(', source)

  for func_name in dependent_funcs:
    dependent_func = getattr(module, func_name, None)
    if callable(dependent_func):
      source += get_source_with_dependencies(dependent_func, visited)
  return source


# check if Metal is supported
def is_metal_supported():
  # first check if we are on macOS
  if platform.system() != 'Darwin':
    return False
  if platform.processor() != 'arm':
    return False
  return True


### VARIABLES ###
home_path = get_home_dir()
kernels_aot_path = os.path.join(home_path, '.brainpy', 'kernels')
is_metal_device = is_metal_supported()


# check if a kernel exists in the database
def _check_kernel_exist(source_md5_encode: str) -> bool:
  # get the realpath of the kernel
  kernel_path = os.path.join(kernels_aot_path, source_md5_encode)

  # check whether the kernel exists
  if os.path.exists(kernel_path):
    return True
  else:
    return False


### KERNEL AOT BUILD ###


def _array_to_field(dtype, shape) -> Any:
  ti = import_taichi()
  if dtype == np.bool_:
    dtype = bool
  elif dtype == np.int8:
    dtype = ti.int8
  elif dtype == np.int16:
    dtype = ti.int16
  elif dtype == np.int32:
    dtype = ti.int32
  elif dtype == np.int64:
    dtype = ti.int64
  elif dtype == np.uint8:
    dtype = ti.uint8
  elif dtype == np.uint16:
    dtype = ti.uint16
  elif dtype == np.uint32:
    dtype = ti.uint32
  elif dtype == np.uint64:
    dtype = ti.uint64
  elif dtype == np.float16:
    dtype = ti.float16
  elif dtype == np.float32:
    dtype = ti.float32
  elif dtype == np.float64:
    dtype = ti.float64
  else:
    raise NotImplementedError(f'Currently we do not support dtype {dtype} in Taichi. '
                              f'If you think it is necessary, please open an issue at '
                              f'https://github.com/brainpy/BrainPy/issues/new')
  return ti.field(dtype=dtype, shape=shape)


# build aot kernel
def _build_kernel(
    source_md5_encode: str,
    kernel: callable,
    ins: dict,
    outs: dict,
    device: str
):
  ti = import_taichi()

  # init arch
  if device == 'cpu':
    if is_metal_device:
      arch = ti.arm64
      device = 'arm64'
    else:
      arch = ti.x64
  elif device == 'gpu':
    arch = ti.cuda
  else:
    raise ValueError(f'Unknown device: {device}')
  with contextlib.redirect_stdout(io.StringIO()):
    ti.init(arch=arch)

  # check arch is available
  if ti.lang.impl.current_cfg().arch != arch:
    raise RuntimeError(f"Arch {arch} is not available")

  # get kernel name
  kernel_name = kernel.__name__

  # replace the name of the func
  kernel.__name__ = f'taichi_kernel_{device}'

  # init template_args_dict
  template_args_dict = {}
  for key, value in ins.items():
    template_args_dict[key] = _array_to_field(value[0], value[1])
  for key, value in outs.items():
    template_args_dict[key] = _array_to_field(value[0], value[1])

  # make aot dir
  kernel_path = os.path.join(kernels_aot_path, source_md5_encode)
  os.makedirs(kernel_path, exist_ok=True)

  # compile kernel
  mod = ti.aot.Module(arch)
  mod.add_kernel(kernel, template_args=template_args_dict)
  mod.save(kernel_path)

  # rename kernel name
  kernel.__name__ = kernel_name


### KERNEL CALL PREPROCESS ###

# convert type to number
type_number_map = {
  int: 0,
  float: 1,
  bool: 2,
  np.dtype('int32'): 0,
  np.dtype('float32'): 1,
  np.dtype('bool'): 2,
  np.dtype('uint8'): 3,
  np.dtype('uint16'): 4,
  np.dtype('uint32'): 5,
  np.dtype('uint64'): 6,
  np.dtype('int8'): 7,
  np.dtype('int16'): 8,
  np.dtype('int64'): 9,
  np.dtype('float16'): 10,
  np.dtype('float64'): 11,
}


# preprocess kernel call cpu
def _preprocess_kernel_call_cpu(
    source_md5_encode: str,
    ins: Sequence,
    outs: Sequence,
) -> list:
  in_out_info = []
  max_dim_count = 0
  for value in ins:
    if value.ndim > max_dim_count:
      max_dim_count = value.ndim

  for value in outs:
    if value.ndim > max_dim_count:
      max_dim_count = value.ndim

  # kernel_path
  kernel_path = os.path.join(kernels_aot_path, source_md5_encode)
  kernel_path = bytes(kernel_path, encoding='utf-8') + b'\0'
  kernel_path = np.array(list(kernel_path), dtype=np.uint8)

  # other args
  in_out_num = np.array([len(ins), len(outs), kernel_path.size], dtype=np.uint32)
  in_out_type_list = np.zeros((len(ins) + len(outs),), dtype=np.uint32)
  in_out_dim_count_list = np.zeros((len(ins) + len(outs),), dtype=np.uint32)
  in_out_elem_count_list = np.zeros((len(ins) + len(outs),), dtype=np.uint32)
  in_out_shape_list = np.zeros((len(ins) + len(outs), max_dim_count), dtype=np.uint32)

  for i, value in enumerate(ins):
    in_out_type_list[i] = type_number_map[value.dtype]
    in_out_dim_count_list[i] = value.ndim
    in_out_elem_count_list[i] = value.size
    for j, dim in enumerate(value.shape):
      in_out_shape_list[i, j] = dim

  b = len(ins)
  for i, value in enumerate(outs):
    in_out_type_list[i + b] = type_number_map[value.dtype]
    in_out_dim_count_list[i + b] = value.ndim
    in_out_elem_count_list[i + b] = value.size
    for j, dim in enumerate(value.shape):
      in_out_shape_list[i + b, j] = dim

  in_out_info.append(in_out_num)
  in_out_info.append(in_out_type_list)
  in_out_info.append(in_out_dim_count_list)
  in_out_info.append(in_out_elem_count_list)
  in_out_info.append(in_out_shape_list)
  in_out_info.append(kernel_path)

  return in_out_info


def _preprocess_kernel_call_gpu(
    source_md5_encode: str,
    ins: dict,
    outs: dict,
) -> bytes:
  if len(ins) + len(outs) > 8:
    raise ValueError('The number of ins and outs must be less than 8!')
  kernel_path = os.path.join(kernels_aot_path, source_md5_encode)

  # other args
  in_out_num = [len(ins), len(outs)]
  in_out_type_list = [0] * 8
  in_out_dim_count_list = [0] * 8
  in_out_elem_count_list = [0] * 8
  in_out_shape_list = [0] * 64

  for i, value in enumerate(ins.values()):
    in_out_type_list[i] = type_number_map[value[0]]
    in_out_dim_count_list[i] = len(value[1])
    in_out_elem_count_list[i] = reduce(lambda x, y: x * y, value[1])
    for j, dim in enumerate(value[1]):
      in_out_shape_list[i * 8 + j] = dim

  for i, value in enumerate(outs.values()):
    in_out_type_list[i + len(ins)] = type_number_map[value[0]]
    in_out_dim_count_list[i + len(ins)] = len(value[1])
    in_out_elem_count_list[i + len(ins)] = reduce(lambda x, y: x * y, value[1])
    for j, dim in enumerate(value[1]):
      in_out_shape_list[(i + len(ins)) * 8 + j] = dim

  # covert to string
  in_out_num_str = ",".join(str(i) for i in in_out_num)
  in_out_type_list_str = ",".join(str(i) for i in in_out_type_list)
  in_out_dim_count_list_str = ",".join(str(i) for i in in_out_dim_count_list)
  in_out_elem_count_list_str = ",".join(str(i) for i in in_out_elem_count_list)
  in_out_shape_list_str = ",".join(str(i) for i in in_out_shape_list)

  opaque = (bytes(in_out_num_str, encoding='utf-8') + b';' +
            bytes(in_out_type_list_str, encoding='utf-8') + b';' +
            bytes(in_out_dim_count_list_str, encoding='utf-8') + b';' +
            bytes(in_out_elem_count_list_str, encoding='utf-8') + b';' +
            bytes(in_out_shape_list_str, encoding='utf-8') + b';' +
            bytes(kernel_path, encoding='utf-8'))

  return opaque


def _XlaOp_to_ShapedArray(c, xla_op):
  xla_op = c.get_shape(xla_op)
  return jax.core.ShapedArray(xla_op.dimensions(), xla_op.element_type())


def _kernel_to_code(kernel, abs_ins, abs_outs, platform):
  codes = f'[taichi {platform} kernel]\n' + get_source_with_dependencies(kernel)
  codes += '\n[ins]: {}'.format("-".join([f'{v.dtype}[{v.shape}]' for v in abs_ins]))
  codes += '\n[outs]: {}'.format("-".join([f'{v.dtype}[{v.shape}]' for v in abs_outs]))
  return codes


def _compile_kernel(kernel, c, platform, *ins, **kwargs):
  # input and output abstract information
  abs_outs = kwargs['outs']
  abs_ins = [_XlaOp_to_ShapedArray(c, v) for v in ins]

  # kernel to code
  codes = _kernel_to_code(kernel, abs_ins, abs_outs, platform)
  source_md5_encode = os.path.join(kernel.__name__, encode_md5(codes))

  # create ins, outs dict from kernel's args
  in_num = len(ins)
  names = tuple(inspect.signature(kernel).parameters.keys())
  in_names, out_names = names[:in_num], names[in_num:]
  ins_dict = {key: (abs_ins[i].dtype, abs_ins[i].shape) for i, key in enumerate(in_names)}
  outs_dict = {key: (abs_outs[i].dtype, abs_outs[i].shape) for i, key in enumerate(out_names)}

  # build kernels
  if not _check_kernel_exist(source_md5_encode):  # TODO: more checking
    try:
      _build_kernel(source_md5_encode, kernel, ins_dict, outs_dict, platform)
    except Exception as e:
      try:
        os.removedirs(os.path.join(kernels_aot_path, source_md5_encode))
      except Exception:
          raise RuntimeError(f'Failed to preprocess info to build kernel:\n\n {codes}') from e
      raise RuntimeError(f'Failed to build kernel:\n\n {codes}') from e

  # returns
  if platform in ['gpu', 'cuda']:
    import_brainpylib_gpu_ops()
    opaque = _preprocess_kernel_call_gpu(source_md5_encode, ins_dict, outs_dict)
    return opaque
  elif platform == 'cpu':
    import_brainpylib_cpu_ops()
    in_out_info = _preprocess_kernel_call_cpu(source_md5_encode, abs_ins, abs_outs)
    return in_out_info
  else:
    raise ValueError(f'Unknown platform: {platform}')


def _taichi_cpu_translation_rule(kernel, c, *ins, **kwargs):
  in_out_info = _compile_kernel(kernel, c, 'cpu', *ins, **kwargs)
  ins = [xla_client.ops.Constant(c, v) for v in in_out_info] + list(ins)
  if is_metal_device:
    fn = b'taichi_kernel_aot_call_cpu_arm64'
  else:
    fn = b'taichi_kernel_aot_call_cpu'

  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=ins,
    operand_shapes_with_layout=tuple(c.get_shape(value) for value in ins),
    shape_with_layout=xla_client.Shape.tuple_shape(
      [xla_client.Shape.array_shape(value.dtype, value.shape, _shape_to_layout(value.shape))
       for value in kwargs['outs']]
    ),
  )


def _taichi_gpu_translation_rule(kernel, c, *ins, **kwargs):
  opaque = _compile_kernel(kernel, c, 'gpu', *ins, **kwargs)
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'taichi_kernel_aot_call_gpu',
    operands=ins,
    operand_shapes_with_layout=tuple(c.get_shape(value) for value in ins),
    shape_with_layout=xla_client.Shape.tuple_shape(
      [xla_client.Shape.array_shape(value.dtype, value.shape, _shape_to_layout(value.shape))
       for value in kwargs['outs']]
    ),
    opaque=opaque,
  )


def register_taichi_cpu_translation_rule(primitive, cpu_kernel):
  xla.backend_specific_translations['cpu'][primitive] = partial(_taichi_cpu_translation_rule, cpu_kernel)


def register_taichi_gpu_translation_rule(primitive, gpu_kernel):
  xla.backend_specific_translations['gpu'][primitive] = partial(_taichi_gpu_translation_rule, gpu_kernel)
