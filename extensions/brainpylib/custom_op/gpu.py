# -*- coding: utf-8 -*-

import numba
import numpy as np
from jax.abstract_arrays import ShapedArray
from jax.lib import xla_client

from .cuda import *

_lambda_no = 0
ctypes.pythonapi.PyCapsule_New.argtypes = [
  ctypes.c_void_p,  # void* pointer
  ctypes.c_char_p,  # const char *name
  ctypes.c_void_p,  # PyCapsule_Destructor destructor
]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object


def _compile_gpu_signature(func, input_dtypes, input_shapes,
                           output_dtypes, output_shapes):
  input_byte_size = tuple(
    np.prod(shape) * dtype.itemsize
    for (shape, dtype) in zip(input_shapes, input_dtypes)
  )
  output_byte_size = tuple(
    np.prod(shape) * dtype.itemsize
    for (shape, dtype) in zip(output_shapes, output_dtypes)
  )

  code_scope = dict(
    func_to_call=func,
    input_shapes=input_shapes,
    input_dtypes=input_dtypes,
    output_shapes=output_shapes,
    output_dtypes=output_dtypes,
    empty=np.empty,
    input_byte_size=input_byte_size,
    output_byte_size=output_byte_size,
    cuMemcpyAsync=cuMemcpyAsync,
    cuStreamSynchronize=cuStreamSynchronize,
    memcpyHostToHost=memcpyHostToHost,
    memcpyHostToDevice=memcpyHostToDevice,
    memcpyDeviceToHost=memcpyDeviceToHost,
    memcpyDeviceToDevice=memcpyDeviceToDevice,
    n_in=len(input_shapes),
  )

  args_in = [
    f'empty(input_shapes[{i}], dtype=input_dtypes[{i}])'
    for i in range(len(input_shapes))
  ]
  cuMemcpyAsync_in = [
    f'cuMemcpyAsync(args_in[{i}].ctypes.data, inout_gpu_ptrs[{i}], input_byte_size[{i}], memcpyDeviceToHost, stream)'
    for i in range(len(input_shapes))
  ]
  args_out = [
    f'empty(output_shapes[{i}], dtype=output_dtypes[{i}])'
    for i in range(len(output_shapes))
  ]
  cuMemcpyAsync_out = [
    f'cuMemcpyAsync(inout_gpu_ptrs[n_in + {i}], args_out[{i}].ctypes.data, output_byte_size[{i}], ' \
    f'memcpyHostToDevice, stream)'
    for i in range(len(output_shapes))
  ]

  code_string = '''
def xla_gpu_custom_call_target(stream, inout_gpu_ptrs, opaque, opaque_len):
  args_out = (
    {args_out}
  )
  args_in = (
    {args_in}
  )
  {cuMemcpyAsync_in}
  cuStreamSynchronize(stream)
  func_to_call(args_out, args_in)
  {cuMemcpyAsync_out}
    '''.format(args_in=",\n    ".join(args_in),
               args_out=",\n    ".join(args_out),
               cuMemcpyAsync_in="\n  ".join(cuMemcpyAsync_in),
               cuMemcpyAsync_out="\n  ".join(cuMemcpyAsync_out))
  # print(code_string)
  exec(compile(code_string.strip(), '', 'exec'), code_scope)

  new_f = code_scope['xla_gpu_custom_call_target']
  wrapper = numba.cfunc(types.void(
    types.voidptr,
    types.CPointer(types.voidptr),
    types.voidptr, types.uint64))
  xla_c_rule = wrapper(new_f)
  target_name = xla_c_rule.native_name.encode("ascii")
  capsule = ctypes.pythonapi.PyCapsule_New(
    xla_c_rule.address,  # A CFFI pointer to a function
    b"xla._CUSTOM_CALL_TARGET",  # A binary string
    None  # PyCapsule object run at destruction
  )
  xla_client.register_custom_call_target(target_name, capsule, "gpu")
  return target_name


def func_gpu_translation(func, abs_eval_fn, c, *inputs):
  if not numba_cffi_loaded:
    raise RuntimeError("Numba cffi could not be loaded.")

  input_shapes = [c.get_shape(arg) for arg in inputs]
  input_dtypes = tuple(shape.element_type() for shape in input_shapes)
  input_dimensions = tuple(shape.dimensions() for shape in input_shapes)
  output_abstract_arrays = abs_eval_fn(*tuple(ShapedArray(shape.dimensions(), shape.element_type())
                                              for shape in input_shapes))
  output_shapes = tuple(array.shape for array in output_abstract_arrays)
  output_dtypes = tuple(array.dtype for array in output_abstract_arrays)
  output_layouts = map(lambda shape: range(len(shape) - 1, -1, -1), output_shapes)
  xla_output_shapes = [xla_client.Shape.array_shape(*arg)
                       for arg in zip(output_dtypes, output_shapes, output_layouts)]
  xla_output_shape = xla_client.Shape.tuple_shape(xla_output_shapes)
  target_name = _compile_gpu_signature(func,
                                       input_dtypes, input_dimensions,
                                       output_dtypes, output_shapes)

  return xla_client.ops.CustomCallWithLayout(
    c,
    target_name,
    operands=inputs,
    operand_shapes_with_layout=input_shapes,
    shape_with_layout=xla_output_shape,
  )
