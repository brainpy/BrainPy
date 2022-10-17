# -*- coding: utf-8 -*-

import ctypes

import numpy as np
from jax.abstract_arrays import ShapedArray
from jax.lib import xla_client
from jax import dtypes
from numba import types, carray, cfunc

_lambda_no = 0
ctypes.pythonapi.PyCapsule_New.argtypes = [
  ctypes.c_void_p,  # void* pointer
  ctypes.c_char_p,  # const char *name
  ctypes.c_void_p,  # PyCapsule_Destructor destructor
]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object


def _compile_cpu_signature(func, input_dtypes, input_shapes,
                           output_dtypes, output_shapes, debug=True):
  code_scope = dict(
    func_to_call=func,
    input_shapes=input_shapes,
    input_dtypes=input_dtypes,
    output_shapes=output_shapes,
    output_dtypes=output_dtypes,
    carray=carray,
  )

  args_in = [
    f'carray(input_ptrs[{i}], input_shapes[{i}], dtype=input_dtypes[{i}])'
    for i in range(len(input_shapes))
  ]
  args_out = [
    f'carray(output_ptrs[{i}], output_shapes[{i}], dtype=output_dtypes[{i}])'
    for i in range(len(output_shapes))
  ]

  code_string = '''
def xla_cpu_custom_call_target(output_ptrs, input_ptrs):
  args_out = (
    {args_out}
  )
  args_in = (
    {args_in}
  )
  func_to_call(args_out, args_in)
    '''.format(args_in=",\n    ".join(args_in),
               args_out=",\n    ".join(args_out))
  if debug: print(code_string)
  exec(compile(code_string.strip(), '', 'exec'), code_scope)

  new_f = code_scope['xla_cpu_custom_call_target']
  xla_c_rule = cfunc(types.void(types.CPointer(types.voidptr),
                                      types.CPointer(types.voidptr)))(new_f)
  target_name = xla_c_rule.native_name.encode("ascii")
  capsule = ctypes.pythonapi.PyCapsule_New(
    xla_c_rule.address,  # A CFFI pointer to a function
    b"xla._CUSTOM_CALL_TARGET",  # A binary string
    None  # PyCapsule object run at destruction
  )
  xla_client.register_custom_call_target(target_name, capsule, "cpu")
  return target_name


def func_cpu_translation(func, abs_eval_fn, c, *inputs, **info):
  input_shapes = [c.get_shape(arg) for arg in inputs]
  for v in info.values():
    if not isinstance(v, (int, float)):
      raise TypeError
    input_shapes.append(xla_client.Shape.array_shape(dtypes.canonicalize_dtype(type(v)), (), ()))
  input_shapes = tuple(input_shapes)
  input_dtypes = tuple(shape.element_type() for shape in input_shapes)
  input_dimensions = tuple(shape.dimensions() for shape in input_shapes)
  output_abstract_arrays = abs_eval_fn(*input_shapes[:len(inputs)], **info)
  output_shapes = tuple(array.shape for array in output_abstract_arrays)
  output_dtypes = tuple(array.dtype for array in output_abstract_arrays)
  output_layouts = map(lambda shape: range(len(shape) - 1, -1, -1), output_shapes)
  xla_output_shapes = [xla_client.Shape.array_shape(*arg)
                       for arg in zip(output_dtypes, output_shapes, output_layouts)]
  xla_output_shape = xla_client.Shape.tuple_shape(xla_output_shapes)
  target_name = _compile_cpu_signature(func,
                                       input_dtypes, input_dimensions,
                                       output_dtypes, output_shapes)

  return xla_client.ops.CustomCallWithLayout(
    c,
    target_name,
    operands=inputs + tuple(xla_client.ops.ConstantLiteral(c, i) for i in info.values()),
    operand_shapes_with_layout=input_shapes,
    shape_with_layout=xla_output_shape,
  )
