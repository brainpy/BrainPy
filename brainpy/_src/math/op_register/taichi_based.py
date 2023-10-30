from functools import partial
import hashlib
import inspect
import pathlib
import sqlite3
from typing import Any
import os

from brainpy._src.math.interoperability import as_jax
from jax.interpreters import xla
from jax.lib import xla_client
import jaxlib.xla_extension
import jax.core
import jax.numpy as jnp
import numpy as np
import taichi as ti

try: 
  from brainpylib import cpu_ops
except:
  cpu_ops = None

try:
  from brainpylib import gpu_ops
except:
  gpu_ops = None


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

### VARIABLES ###
home_path = get_home_dir()
db_path = os.path.join(home_path, '.brainpy', 'kernels.db')
kernels_aot_path = os.path.join(home_path, '.brainpy', 'kernels')

### DATABASE ###

# initialize the database
def init_database():
  if not os.path.exists(os.path.join(home_path, '.brainpy')):
    os.mkdir(os.path.join(home_path, '.brainpy'))
    print('Create .brainpy directory')
  if os.path.exists(db_path):
    if os.path.exists(kernels_aot_path):
      return
    else:
      os.mkdir(kernels_aot_path)
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

# jnp dtype to taichi type
type_map4template = {
        jnp.dtype("bool"): bool,
        jnp.dtype("int8"): ti.int8,
        jnp.dtype("int16"): ti.int16,
        jnp.dtype("int32"): ti.int32,
        jnp.dtype("int64"): ti.int64,
        jnp.dtype("uint8"): ti.uint8,
        jnp.dtype("uint16"): ti.uint16,
        jnp.dtype("uint32"): ti.uint32,
        jnp.dtype("uint64"): ti.uint64,
        jnp.dtype("float16"): ti.float16,
        jnp.dtype("float32"): ti.float32,
        jnp.dtype("float64"): ti.float64,
}

def jnp_array2taichi_field(obj: Any) -> Any:
  if isinstance(obj, jnp.ndarray):
    return ti.field(dtype=type_map4template[obj.dtype], shape=obj.shape)
  elif isinstance(obj, jax.core.ShapedArray):
    return ti.field(dtype=type_map4template[obj.dtype], shape=obj.shape)
  elif isinstance(obj, jaxlib.xla_extension.XlaOp):
    return ti.field(dtype=type_map4template[obj.dtype], shape=obj.shape)
  else:
    raise TypeError(f"{obj} is not a jnp.ndarray")
    
# build aot kernel
def build_kernel(
    source_md5_encode: str,
    kernel: callable,
    ins: dict,
    outs: dict,
    device: str
):
  #init arch
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

  # init kernel
  # kernel = ti.kernel(kernel)

  # init template_args_dict
  template_args_dict = {}
  for key, value in ins.items():
    template_args_dict[key] = jnp_array2taichi_field(value)
  for key, value in outs.items():
    template_args_dict[key] = jnp_array2taichi_field(value)

  # make aot dir
  kernel_path = os.path.join(kernels_aot_path, source_md5_encode)
  os.makedirs(kernel_path, exist_ok=True)

  # compile kernel
  mod = ti.aot.Module(arch)
  mod.add_kernel(kernel, template_args=template_args_dict)
  mod.save(kernel_path)

### KER NEL CALL PREPROCESS ###

# convert type to number
type_number_map = {
    int: 0,
    float: 1,
    bool: 2,
    ti.int32: 0,
    ti.float32: 1,
    ti.u8: 3,
    ti.u16: 4,
    ti.u32: 5,
    ti.u64: 6,
    ti.i8: 7,
    ti.i16: 8,
    ti.i64: 9,
    ti.f16: 10,
    ti.f64: 11,
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
    ins: dict,
    outs: dict,
)-> list:
  ins_list = []
  max_dim_count = 0
  for value in ins.values():
    if value.ndim > max_dim_count:
      max_dim_count = value.ndim
  
  for value in outs.values():
    if value.ndim > max_dim_count:
      max_dim_count = value.ndim

  # kernel_path
  kernel_path = os.path.join(kernels_aot_path, source_md5_encode)
  kernel_path = bytes(kernels_aot_path, encoding='utf-8') + b'\0'
  kernel_path = jnp.array(list(kernel_path), dtype=jnp.uint8)

  # other args
  in_out_num = jnp.array([len(ins), len(outs), kernel_path.size], dtype=jnp.uint32)
  in_out_type_list = jnp.zeros((len(ins) + len(outs),), dtype=jnp.uint8)
  in_out_dim_count_list = jnp.zeros((len(ins) + len(outs),), dtype=jnp.uint8)
  in_out_elem_count_list = jnp.zeros((len(ins) + len(outs),), dtype=jnp.uint32)
  in_out_shape_list = jnp.zeros((len(ins) + len(outs), max_dim_count), dtype=jnp.uint32)

  for i, value in enumerate(ins.values()):
    in_out_type_list = in_out_type_list.at[i].set(type_number_map[value.dtype])
    in_out_dim_count_list = in_out_dim_count_list.at[i].set(value.ndim)
    in_out_elem_count_list = in_out_elem_count_list.at[i].set(value.size)
    for j, dim in enumerate(value.shape):
      in_out_shape_list = in_out_shape_list.at[i, j].set(dim)

  for i, value in enumerate(outs.values()):
    in_out_type_list = in_out_type_list.at[i + len(ins)].set(type_number_map[value.dtype])
    in_out_dim_count_list = in_out_dim_count_list.at[i + len(ins)].set(value.ndim)
    in_out_elem_count_list = in_out_elem_count_list.at[i + len(ins)].set(value.size)
    for j, dim in enumerate(value.shape):
      in_out_shape_list = in_out_shape_list.at[i + len(ins), j].set(dim)

  ins_list.append(as_jax(in_out_num))
  ins_list.append(as_jax(in_out_type_list))
  ins_list.append(as_jax(in_out_dim_count_list))
  ins_list.append(as_jax(in_out_elem_count_list))
  ins_list.append(as_jax(in_out_shape_list))
  ins_list.append(as_jax(kernel_path))

  for value in ins.values():
    ins_list.append(value)

  return ins_list

def preprocess_kernel_call_gpu(
    source_md5_encode: str,
    ins: dict,
    outs: dict,
)-> bytes:
  
  if len(ins) + len(outs) > 8:
            raise ValueError('The number of ins and outs must be less than 8!')
        # set ins's array to jax by using as_jax
  ins_list = []

  kernel_path = os.path.join(kernels_aot_path, source_md5_encode)

  # other args
  in_out_num = [len(ins), len(outs)]
  in_out_type_list = [0] * 8
  in_out_dim_count_list = [0] * 8
  in_out_elem_count_list = [0] * 8
  in_out_shape_list = [0] * 64

  for i, value in enumerate(ins.values()):
    in_out_type_list[i] = type_number_map[value.dtype]
    in_out_dim_count_list[i] = value.ndim
    in_out_elem_count_list[i] = value.size
    for j, dim in enumerate(value.shape):
      in_out_shape_list[i * 8 + j] = dim

  for i, value in enumerate(outs.values()):
    in_out_type_list[i + len(ins)] = type_number_map[value.dtype]
    in_out_dim_count_list[i + len(ins)] = value.ndim
    in_out_elem_count_list[i + len(ins)] = value.size
    for j, dim in enumerate(value.shape):
      in_out_shape_list[(i + len(ins)) * 8 + j] = dim

  # covert to string
  in_out_num_str = ",".join(str(i) for i in in_out_num)
  in_out_type_list_str = ",".join(str(i) for i in in_out_type_list)
  in_out_dim_count_list_str = ",".join(str(i) for i in in_out_dim_count_list)
  in_out_elem_count_list_str = ",".join(str(i) for i in in_out_elem_count_list)
  in_out_shape_list_str = ",".join(str(i) for i in in_out_shape_list)

  opaque = bytes(in_out_num_str, encoding='utf-8') + b';' + \
           bytes(in_out_type_list_str, encoding='utf-8') + b';' + \
           bytes(in_out_dim_count_list_str, encoding='utf-8') + b';' + \
           bytes(in_out_elem_count_list_str, encoding='utf-8') + b';' + \
           bytes(in_out_shape_list_str, encoding='utf-8') + b';' + \
           bytes(kernel_path, encoding='utf-8')
  
  return opaque


def _taichi_cpu_translation_rule(prim, kernel, c, *ins):
  outs = prim.abstract_eval()[0]

  init_database()
  # find the path of taichi in python site_packages
  taichi_path = ti.__path__[0]
  taichi_c_api_install_dir = os.path.join(taichi_path, '_lib', 'c_api')

  os.environ.update({
    'TAICHI_C_API_INSTALL_DIR': taichi_c_api_install_dir,
    'TI_LIB_DIR': os.path.join(taichi_c_api_install_dir, 'runtime')
  })

  source_md5_encode = encode_md5('cpu' + inspect.getsource(kernel) + str([c.get_shape(value) for value in ins]) + str([value.shape for value in outs]))

  # create ins, outs dict from kernel's args
  ins_dict = {}
  outs_dict = {}
  in_num = len(ins)
  out_num = len(outs)

  params = inspect.signature(kernel).parameters
  for i, (name, _) in enumerate(params.items()):
    if i < in_num:
      ins_dict[name] = ins[i]
    else:
      outs_dict[name] = outs[i - in_num]

  if(not check_kernel_exist(source_md5_encode)):
    try:
      build_kernel(source_md5_encode, kernel, ins_dict, outs_dict, 'cpu')
    except:
      raise RuntimeError('Failed to build kernel')
  
  ins_list = preprocess_kernel_call_cpu(source_md5_encode, ins_dict, outs_dict)

  fn = b'taichi_kernel_call_cpu'
  operands = ins_list
  operands_shapes_with_layout = tuple(c.get_shape(value) for value in operands)
  shape_with_layout = xla_client.Shape.tuple_shape(
    [xla_client.Shape.array_shape(value.dtype, value.shape, tuple(range(value.dim)))] for value in outs
  )

  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=operands,
    operand_shapes_with_layout=operands_shapes_with_layout,
    shape_with_layout=shape_with_layout,
  )

def _taichi_gpu_translation_rule(prim, kernel, c, *ins):
  outs = prim.abstract_eval()[0]

  init_database()
  # find the path of taichi in python site_packages
  taichi_path = ti.__path__[0]
  taichi_c_api_install_dir = os.path.join(taichi_path, '_lib', 'c_api')

  os.environ.update({
    'TAICHI_C_API_INSTALL_DIR': taichi_c_api_install_dir,
    'TI_LIB_DIR': os.path.join(taichi_c_api_install_dir, 'runtime')
  })

  source_md5_encode = encode_md5('cpu' + inspect.getsource(kernel) + str([value.shape for value in ins]) + str([value.shape for value in outs]))

  # create ins, outs dict from kernel's args
  ins_dict = {}
  outs_dict = {}
  in_num = len(ins)
  out_num = len(outs)

  params = inspect.signature(kernel).parameters
  for i, (name, _) in enumerate(params.items()):
    if i < in_num:
      ins_dict[name] = ins[i]
    else:
      outs_dict[name] = outs[i - in_num]

  if(not check_kernel_exist(source_md5_encode)):
    try:
      build_kernel(source_md5_encode, kernel, ins_dict, outs_dict, 'gpu')
    except:
      raise RuntimeError('Failed to build kernel')
    
  opaque = preprocess_kernel_call_gpu(source_md5_encode, ins_dict, outs_dict)

  fn = b'taichi_kernel_call_gpu'

  operands = ins
  operands_shapes_with_layout = tuple(c.get_shape(value) for value in operands)
  shape_with_layout = xla_client.Shape.tuple_shape(
    [xla_client.Shape.array_shape(value.dtype, value.shape, tuple(range(value.dim)))] for value in outs
  )

  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=operands,
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
