from functools import partial, reduce
from typing import List

import jax
import numpy as np
from jax.interpreters import xla, mlir
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

from brainpy._src.dependency_check import (import_cupy,
                                           import_brainpylib_gpu_ops)
from brainpy._src.math.op_register.utils import _shape_to_layout
from brainpy.errors import PackageMissingError

cp = import_cupy(error_if_not_found=False)

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


def _preprocess_kernel_call_gpu(
    grid: int,
    block: int,
    func_ptr: int,
    shared_mem: int,
    *ins,
    outs: List[jax.ShapeDtypeStruct],
):
  grid = (grid + (1, 1))[:3]
  block = (block + (1, 1))[:3]
  in_num = len(ins)
  out_num = len(outs)
  in_out_num = [in_num, out_num]

  out_type_list = [0] * out_num
  out_elem_count_list = [0] * out_num

  for i, value in enumerate(outs):
    out_type_list[i] = type_number_map[value.dtype]
    out_elem_count_list[i] = reduce(lambda x, y: x * y, value.shape)

  grid = ",".join(str(i) for i in grid)
  block = ",".join(str(i) for i in block)
  in_out_num_str = ",".join(str(i) for i in in_out_num)
  out_type_list_str = ",".join(str(i) for i in out_type_list)
  out_elem_count_list_str = ",".join(str(i) for i in out_elem_count_list)

  opaque = (bytes(str(func_ptr), encoding='utf-8') + b';' +
            bytes(str(shared_mem), encoding='utf-8') + b';' +
            bytes(in_out_num_str, encoding='utf-8') + b';' +
            bytes(grid, encoding='utf-8') + b';' +
            bytes(block, encoding='utf-8') + b';' +
            bytes(out_type_list_str, encoding='utf-8') + b';' +
            bytes(out_elem_count_list_str, encoding='utf-8') + b';')
  return opaque


def _cupy_xla_gpu_translation_rule(kernel, c, *ins, **kwargs):
  grid = kwargs.get('grid', None)
  block = kwargs.get('block', None)
  shared_mem = kwargs.get('shared_mem', 0)
  if grid is None or block is None:
    raise ValueError('The grid and block should be specified for the cupy kernel.')

  # compile
  mod = cp.RawModule(code=kernel)
  try:
    kernel_func = mod.get_function('kernel')
  except AttributeError:
    raise ValueError('The \'kernel\' function is not found in the module.')

  # preprocess
  import_brainpylib_gpu_ops()
  opaque = _preprocess_kernel_call_gpu(grid, block, kernel_func.kernel.ptr, shared_mem, *ins, outs=kwargs['outs'])

  # create custom call
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'cupy_kernel_call_gpu',
    operands=ins,
    operand_shapes_with_layout=tuple(c.get_shape(value) for value in ins),
    shape_with_layout=xla_client.Shape.tuple_shape(
      [xla_client.Shape.array_shape(value.dtype, value.shape, _shape_to_layout(value.shape))
       for value in kwargs['outs']]
    ),
    opaque=opaque,
  )


def register_cupy_xla_gpu_translation_rule(primitive, gpu_kernel):
  xla.backend_specific_translations['gpu'][primitive] = partial(_cupy_xla_gpu_translation_rule, gpu_kernel)


def _cupy_mlir_gpu_translation_rule(kernel, c, *ins, **kwargs):
  grid = kwargs.get('grid', None)
  block = kwargs.get('block', None)
  shared_mem = kwargs.get('shared_mem', 0)
  if grid is None or block is None:
    raise ValueError('The grid and block should be specified for the cupy kernel.')

  # compile
  mod = cp.RawModule(code=kernel)
  try:
    kernel_func = mod.get_function('kernel')
  except AttributeError:
    raise ValueError('The \'kernel\' function is not found in the module.')

  # preprocess
  import_brainpylib_gpu_ops()
  opaque = _preprocess_kernel_call_gpu(grid, block, kernel_func.kernel.ptr, shared_mem, *ins, outs=kwargs['outs'])

  input_layouts = [_shape_to_layout(a.shape) for a in c.avals_in]
  result_types = [mlir.aval_to_ir_type(out) for out in c.avals_out]
  output_layouts = [_shape_to_layout(a.shape) for a in c.avals_out]

  return custom_call(
    call_target_name='cupy_kernel_call_gpu',
    operands=ins,
    operand_layouts=list(input_layouts),
    result_layouts=list(output_layouts),
    result_types=list(result_types),
    backend_config=opaque,
    has_side_effect=False,
  ).results


def register_cupy_mlir_gpu_translation_rule(primitive, gpu_kernel):
  if cp is None:
    raise PackageMissingError("cupy", 'register cupy mlir gpu translation rule')

  rule = partial(_cupy_mlir_gpu_translation_rule, gpu_kernel)
  mlir.register_lowering(primitive, rule, platform='gpu')
