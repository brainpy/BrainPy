import hashlib
import inspect
import os
import pathlib
import re
import sqlite3
from functools import partial, reduce
from typing import Any

import numpy as np
from jax.interpreters import xla
from jax.lib import xla_client

import brainpy.math as bm
from .utils import _shape_to_layout
from ..brainpylib_check import import_taichi


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


### VARIABLES ###
home_path = get_home_dir()
db_path = os.path.join(home_path, '.brainpy', 'kernels.db')
kernels_aot_path = os.path.join(home_path, '.brainpy', 'kernels')


### DATABASE ###

# initialize the database
def init_database():
  if not os.path.exists(os.path.join(home_path, '.brainpy')):
    os.makedirs(os.path.join(home_path, '.brainpy'))
  if os.path.exists(db_path):
    if os.path.exists(kernels_aot_path):
      return
    else:
      os.makedirs(kernels_aot_path)
  else:
    create_database()


# create the database
def create_database():
  # remove the old database
  if os.path.exists(db_path):
    os.remove(db_path)

  # create the new database
  conn = sqlite3.connect(db_path)

  # get the cursor
  c = conn.cursor()

  # create the table
  c.execute('''
    CREATE TABLE kernels (source_md5_encode TEXT PRIMARY KEY)
            ''')
  conn.commit()
  conn.close()


# insert a kernel into the database
def insert(source_md5_encode: str):
  # connect to the database
  conn = sqlite3.connect(db_path)
  c = conn.cursor()

  c.execute('''
    INSERT INTO kernels (source_md5_encode)
    VALUES (?)
    ''', (source_md5_encode,))
  conn.commit()
  conn.close()


# check if a kernel exists in the database
def check_kernel_exist(source_md5_encode: str) -> bool:
  # connect to the database
  conn = sqlite3.connect(db_path)
  c = conn.cursor()

  # check kernel exist
  c.execute('''
    SELECT * FROM kernels WHERE source_md5_encode = ?
    ''', (source_md5_encode,))

  # get result
  result = c.fetchone()
  conn.close()

  if result is None:
    insert(source_md5_encode)
    return False
  else:
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
  elif dtype == np.int8: dtype= ti.int8
  elif dtype == np.int16: dtype= ti.int16
  elif dtype == np.int32: dtype= ti.int32
  elif dtype == np.int64: dtype= ti.int64
  elif dtype == np.uint8: dtype= ti.uint8
  elif dtype == np.uint16: dtype= ti.uint16
  elif dtype == np.uint32: dtype= ti.uint32
  elif dtype == np.uint64: dtype= ti.uint64
  elif dtype == np.float16: dtype= ti.float16
  elif dtype == np.float32: dtype= ti.float32
  elif dtype == np.float64: dtype= ti.float64
  else:
    raise TypeError
  return ti.field(dtype=dtype, shape=shape)


# build aot kernel
def build_kernel(
    source_md5_encode: str,
    kernel: callable,
    ins: dict,
    outs: dict,
    device: str
):
  ti = import_taichi()

  # init arch
  arch = None
  if device == 'cpu':
    arch = ti.x64
  elif device == 'gpu':
    arch = ti.cuda

  ti.init(arch=arch)

  # check arch is available
  if ti.lang.impl.current_cfg().arch != arch:
    raise RuntimeError(f"Arch {arch} is not available")

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
def preprocess_kernel_call_cpu(
    source_md5_encode: str,
    ins: list,
    outs: list,
) -> list:
  ins_list = []
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
  kernel_path = bm.array(list(kernel_path), dtype=bm.uint8)

  # other args
  in_out_num = bm.array([len(ins), len(outs), kernel_path.size], dtype=bm.uint32)
  in_out_type_list = bm.zeros((len(ins) + len(outs),), dtype=bm.uint32)
  in_out_dim_count_list = bm.zeros((len(ins) + len(outs),), dtype=bm.uint32)
  in_out_elem_count_list = bm.zeros((len(ins) + len(outs),), dtype=bm.uint32)
  in_out_shape_list = bm.zeros((len(ins) + len(outs), max_dim_count), dtype=bm.uint32)

  for i, value in enumerate(ins):
    in_out_type_list = in_out_type_list.at[i].set(type_number_map[value.dtype])
    in_out_dim_count_list = in_out_dim_count_list.at[i].set(value.ndim)
    in_out_elem_count_list = in_out_elem_count_list.at[i].set(value.size)
    for j, dim in enumerate(value.shape):
      in_out_shape_list = in_out_shape_list.at[i, j].set(dim)

  for i, value in enumerate(outs):
    in_out_type_list = in_out_type_list.at[i + len(ins)].set(type_number_map[value.dtype])
    in_out_dim_count_list = in_out_dim_count_list.at[i + len(ins)].set(value.ndim)
    in_out_elem_count_list = in_out_elem_count_list.at[i + len(ins)].set(value.size)
    for j, dim in enumerate(value.shape):
      in_out_shape_list = in_out_shape_list.at[i + len(ins), j].set(dim)

  ins_list.append(in_out_num)
  ins_list.append(in_out_type_list)
  ins_list.append(in_out_dim_count_list)
  ins_list.append(in_out_elem_count_list)
  ins_list.append(in_out_shape_list)
  ins_list.append(kernel_path)

  return ins_list


def preprocess_kernel_call_gpu(
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


def _taichi_cpu_translation_rule(prim, kernel, c, *ins):
  outs = prim.abstract_eval()[0]

  output_shapes = tuple(out.shape for out in outs)
  output_dtypes = tuple(out.dtype for out in outs)
  input_layouts = tuple(c.get_shape(arg) for arg in ins)
  input_dtypes = tuple(inp.element_type() for inp in input_layouts)
  input_shapes = tuple(inp.dimensions() for inp in input_layouts)

  init_database()

  # create ins, outs dict from kernel's args
  ins_dict = {}
  outs_dict = {}
  in_num = len(ins) - 6

  params = inspect.signature(kernel).parameters
  for i, (name, _) in enumerate(params.items()):
    if i < in_num:
      ins_dict[name] = (input_dtypes[i + 6], input_shapes[i + 6])
    else:
      outs_dict[name] = (output_dtypes[i - in_num], output_shapes[i - in_num])

  source_md5_encode = encode_md5('cpu' + get_source_with_dependencies(kernel) +
                                 str([(value[0], value[1]) for value in ins_dict.values()]) +
                                 str([(value[0], value[1]) for value in outs_dict.values()]))

  if not check_kernel_exist(source_md5_encode):
    try:
      build_kernel(source_md5_encode, kernel, ins_dict, outs_dict, 'cpu')
    except Exception as e:
      raise RuntimeError('Failed to build kernel') from e

  operands_shapes_with_layout = tuple(c.get_shape(value) for value in ins)
  shape_with_layout = xla_client.Shape.tuple_shape(
    [xla_client.Shape.array_shape(value.dtype, value.shape, _shape_to_layout(value.shape)) for value in outs]
  )

  return xla_client.ops.CustomCallWithLayout(
    c,
    b'taichi_kernel_call_cpu',
    operands=ins,
    operand_shapes_with_layout=operands_shapes_with_layout,
    shape_with_layout=shape_with_layout,
  )


def _taichi_gpu_translation_rule(prim, kernel, c, *ins):
  outs = prim.abstract_eval()[0]

  output_shapes = tuple(out.shape for out in outs)
  output_dtypes = tuple(out.dtype for out in outs)
  input_layouts = tuple(c.get_shape(arg) for arg in ins)
  input_dtypes = tuple(inp.element_type() for inp in input_layouts)
  input_shapes = tuple(inp.dimensions() for inp in input_layouts)

  init_database()

  # create ins, outs dict from kernel's args
  in_num = len(ins)
  params = inspect.signature(kernel).parameters
  names = tuple(params.keys())
  in_names = names[:in_num]
  out_names = names[in_num:]
  ins_dict = {key: (dtype, shape) for key, shape, dtype in zip(in_names, input_shapes, input_dtypes)}
  outs_dict = {key: (dtype, shape) for key, shape, dtype in zip(out_names, output_shapes, output_dtypes)}
  source_md5_encode = encode_md5('gpu' + get_source_with_dependencies(kernel) +
                                 str([(value[0], value[1]) for value in ins_dict.values()]) +
                                 str([(value[0], value[1]) for value in outs_dict.values()]))

  if not check_kernel_exist(source_md5_encode):
    try:
      build_kernel(source_md5_encode, kernel, ins_dict, outs_dict, 'gpu')
    except Exception as e:
      raise RuntimeError('Failed to build Taichi GPU kernel') from e

  opaque = preprocess_kernel_call_gpu(source_md5_encode, ins_dict, outs_dict)

  operands_shapes_with_layout = tuple(c.get_shape(value) for value in ins)
  shape_with_layout = xla_client.Shape.tuple_shape(
    [xla_client.Shape.array_shape(value.dtype, value.shape, _shape_to_layout(value.shape)) for value in outs]
  )

  return xla_client.ops.CustomCallWithLayout(
    c,
    b'taichi_kernel_call_gpu',
    operands=ins,
    operand_shapes_with_layout=operands_shapes_with_layout,
    shape_with_layout=shape_with_layout,
    opaque=opaque,
  )


def register_taichi_cpu_translation_rule(primitive, cpu_kernel):
  xla.backend_specific_translations['cpu'][primitive] = partial(_taichi_cpu_translation_rule,
                                                                primitive, cpu_kernel)


def register_taichi_gpu_translation_rule(primitive, gpu_kernel):
  xla.backend_specific_translations['gpu'][primitive] = partial(_taichi_gpu_translation_rule,
                                                                primitive, gpu_kernel)
