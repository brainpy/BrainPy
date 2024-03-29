from functools import partial, reduce
from typing import List, Tuple

import jax
import numpy as np
from jax.interpreters import xla, mlir
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

from brainpy._src.dependency_check import (import_cupy,
                                           import_cupy_jit,
                                           import_brainpylib_gpu_ops)
from brainpy._src.math.op_register.utils import _shape_to_layout
from brainpy.errors import PackageMissingError

cp = import_cupy(error_if_not_found=False)
cp_jit = import_cupy_jit(error_if_not_found=False)

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
    grid: Tuple[int],
    block: Tuple[int],
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


def _cupy_raw_module_xla_gpu_translation_rule(kernel, c, *ins, **kwargs):
  grid = kwargs.get('grid', None)
  block = kwargs.get('block', None)
  shared_mem = kwargs.get('shared_mem', 0)
  if grid is None or block is None:
    raise ValueError('The grid and block should be specified for the cupy kernel.')

  # preprocess
  import_brainpylib_gpu_ops()
  # THE KEY:
  # - using the kernel pointer at "kernel.kernel.ptr"
  opaque = _preprocess_kernel_call_gpu(grid, block, kernel.kernel.ptr, shared_mem, *ins, outs=kwargs['outs'])

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


def register_cupy_raw_module_xla_gpu_translation_rule(primitive, gpu_kernel):
  xla.backend_specific_translations['gpu'][primitive] = partial(_cupy_raw_module_xla_gpu_translation_rule, gpu_kernel)


def _cupy_raw_module_mlir_gpu_translation_rule(kernel, c, *ins, **kwargs):
  grid = kwargs.get('grid', None)
  block = kwargs.get('block', None)
  shared_mem = kwargs.get('shared_mem', 0)
  if grid is None or block is None:
    raise ValueError('The grid and block should be specified for the cupy kernel.')

  # preprocess
  import_brainpylib_gpu_ops()
  opaque = _preprocess_kernel_call_gpu(grid, block, kernel.kernel.ptr, shared_mem, *ins, outs=kwargs['outs'])

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


def register_cupy_raw_module_mlir_gpu_translation_rule(primitive, gpu_kernel):
  if cp is None:
    raise PackageMissingError("cupy", 'register cupy mlir gpu translation rule')

  rule = partial(_cupy_raw_module_mlir_gpu_translation_rule, gpu_kernel)
  mlir.register_lowering(primitive, rule, platform='gpu')


def _to_cupy_array_or_scalar(dtype, ndim):
  # THE KEY
  # - using the cupy jit compiler to get the type
  if ndim != 0:
    t = cp_jit._cuda_types.CArray(dtype=dtype,
                                  ndim=ndim,
                                  is_c_contiguous=True,
                                  index_32_bits=True)
  else:
    t = cp_jit._cuda_types.Scalar(dtype=dtype)
  return t


def _compile_kernel_xla(kernel, in_types):
  # THE KEY
  # - get the kernel function from the cache
  device_id = cp.cuda.get_device_id()
  kern, enable_cg = kernel._cache.get((in_types, device_id), (None, None))

  if kern is None:
    # THE KEY:
    # - compile the kernel function
    result = kernel._cached_codes.get(in_types)
    if result is None:
      result = cp_jit._compile.transpile(
        kernel._func,
        ['extern "C"', '__global__'],
        'cuda',
        in_types,
        cp_jit._cuda_types.void,
      )
      kernel._cached_codes[in_types] = result
    fname = result.func_name
    enable_cg = result.enable_cooperative_groups
    options = result.options
    backend = result.backend
    if backend == 'nvcc':
      options += ('-DCUPY_JIT_NVCC',)
    jitify = result.jitify
    module = cp._core.core.compile_with_cache(
      source=result.code,
      options=options,
      backend=backend,
      jitify=jitify,
    )
    kern = module.get_function(fname)
    kernel._cache[(in_types, device_id)] = (kern, enable_cg)

  return kern


def get_jit_kernel_xla(kernel, c, *ins, outs):
  # get the input types
  in_types = []
  for x in ins:
    x = c.get_shape(x)
    in_types.append(_to_cupy_array_or_scalar(x.element_type(), len(x.dimensions())))
  for x in outs:
    in_types.append(_to_cupy_array_or_scalar(x.dtype, x.ndim))
  in_types = tuple(in_types)
  # compile the kernel
  return _compile_kernel_xla(kernel, in_types)


def get_jit_kernel_mlir(kernel, c):
  # get the input types
  in_types = []
  for x in c.avals_in:
    in_types.append(_to_cupy_array_or_scalar(x.dtype, x.ndim))
  for x in c.avals_out:
    in_types.append(_to_cupy_array_or_scalar(x.dtype, x.ndim))
  in_types = tuple(in_types)
  # compile the kernel
  return _compile_kernel_xla(kernel, in_types)


def _cupy_jit_kernel_xla_gpu_translation_rule(kernel, c, *ins, **kwargs):
  kernel_func = get_jit_kernel_xla(kernel, c, *ins, outs=kwargs['outs'])
  grid = kwargs.get('grid', None)
  block = kwargs.get('block', None)
  shared_mem = kwargs.get('shared_mem', 0)
  if grid is None or block is None:
    raise ValueError('The grid and block should be specified for the cupy kernel.')

  # preprocess
  import_brainpylib_gpu_ops()
  opaque = _preprocess_kernel_call_gpu(grid, block, kernel_func.ptr, shared_mem, *ins, outs=kwargs['outs'])

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


def register_cupy_jit_kernel_xla_gpu_translation_rule(primitive, gpu_kernel):
  xla.backend_specific_translations['gpu'][primitive] = partial(_cupy_jit_kernel_xla_gpu_translation_rule, gpu_kernel)


def _cupy_jit_kernel_mlir_gpu_translation_rule(kernel, c, *ins, **kwargs):
  kernel_func = get_jit_kernel_mlir(kernel, c)
  grid = kwargs.get('grid', None)
  block = kwargs.get('block', None)
  shared_mem = kwargs.get('shared_mem', 0)
  if grid is None or block is None:
    raise ValueError('The grid and block should be specified for the cupy kernel.')

  # preprocess
  import_brainpylib_gpu_ops()
  opaque = _preprocess_kernel_call_gpu(grid, block, kernel_func.ptr, shared_mem, *ins, outs=kwargs['outs'])

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


def register_cupy_jit_kernel_mlir_gpu_translation_rule(primitive, gpu_kernel):
  if cp is None:
    raise PackageMissingError("cupy", 'register cupy mlir gpu translation rule')

  rule = partial(_cupy_jit_kernel_mlir_gpu_translation_rule, gpu_kernel)
  mlir.register_lowering(primitive, rule, platform='gpu')
