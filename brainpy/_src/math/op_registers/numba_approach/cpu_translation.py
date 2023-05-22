# -*- coding: utf-8 -*-

import ctypes

from jax import dtypes, numpy as jnp
from jax.core import ShapedArray
from jax.lib import xla_client
from numba import types, carray, cfunc

__all__ = [
  'compile_cpu_signature_with_numba'
]

ctypes.pythonapi.PyCapsule_New.argtypes = [
  ctypes.c_void_p,  # void* pointer
  ctypes.c_char_p,  # const char *name
  ctypes.c_void_p,  # PyCapsule_Destructor destructor
]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object


def _cpu_translation(func, abs_eval_fn, multiple_results, c, *inputs, **info):
  target_name, inputs, input_shapes, xla_output_shapes = \
    compile_cpu_signature_with_numba(c, func, abs_eval_fn, multiple_results, inputs, info)
  return xla_client.ops.CustomCallWithLayout(
    c,
    target_name,
    operands=inputs,
    operand_shapes_with_layout=input_shapes,
    shape_with_layout=xla_output_shapes,
  )


def _cpu_signature(
    func,
    input_dtypes,
    input_shapes,
    output_dtypes,
    output_shapes,
    multiple_results: bool,
    debug: bool = False
):
  code_scope = dict(
    func_to_call=func,
    input_shapes=input_shapes,
    input_dtypes=input_dtypes,
    output_shapes=output_shapes,
    output_dtypes=output_dtypes,
    carray=carray,
  )

  # inputs
  if len(input_shapes) > 1:
    args_in = [
      f'carray(input_ptrs[{i}], input_shapes[{i}], dtype=input_dtypes[{i}]),'
      for i in range(len(input_shapes))
    ]
    args_in = '(\n    ' + "\n    ".join(args_in) + '\n  )'
  else:
    args_in = 'carray(input_ptrs[0], input_shapes[0], dtype=input_dtypes[0])'

  # outputs
  if multiple_results:
    args_out = [
      f'carray(output_ptrs[{i}], output_shapes[{i}], dtype=output_dtypes[{i}]),'
      for i in range(len(output_shapes))
    ]
    args_out = '(\n    ' + "\n    ".join(args_out) + '\n  )'
  else:
    args_out = 'carray(output_ptrs, output_shapes[0], dtype=output_dtypes[0])'

  # function body
  code_string = '''
def xla_cpu_custom_call_target(output_ptrs, input_ptrs):
  args_out = {args_out}
  args_in = {args_in}
  func_to_call(args_out, args_in)
    '''.format(args_in=args_in,
               args_out=args_out)
  if debug: print(code_string)
  exec(compile(code_string.strip(), '', 'exec'), code_scope)

  new_f = code_scope['xla_cpu_custom_call_target']
  if multiple_results:
    xla_c_rule = cfunc(types.void(types.CPointer(types.voidptr),
                                  types.CPointer(types.voidptr)))(new_f)
  else:
    xla_c_rule = cfunc(types.void(types.voidptr, types.CPointer(types.voidptr)))(new_f)
  target_name = xla_c_rule.native_name.encode("ascii")
  capsule = ctypes.pythonapi.PyCapsule_New(
    xla_c_rule.address,  # A CFFI pointer to a function
    b"xla._CUSTOM_CALL_TARGET",  # A binary string
    None  # PyCapsule object run at destruction
  )
  xla_client.register_custom_call_target(target_name, capsule, "cpu")
  return target_name


def compile_cpu_signature_with_numba(
    c,
    func,
    abs_eval_fn,
    multiple_results,
    inputs: tuple,
    description: dict = None,
):
  input_layouts = [c.get_shape(arg) for arg in inputs]
  info_inputs = []
  if description is None: description = dict()
  for v in description.values():
    if isinstance(v, (int, float)):
      input_layouts.append(xla_client.Shape.array_shape(dtypes.canonicalize_dtype(type(v)), (), ()))
      info_inputs.append(xla_client.ops.ConstantLiteral(c, v))
    elif isinstance(v, (tuple, list)):
      v = jnp.asarray(v)
      input_layouts.append(xla_client.Shape.array_shape(v.dtype, v.shape, tuple(range(len(v.shape) - 1, -1, -1))))
      info_inputs.append(xla_client.ops.Constant(c, v))
    else:
      raise TypeError
  input_layouts = tuple(input_layouts)
  input_dtypes = tuple(shape.element_type() for shape in input_layouts)
  input_dimensions = tuple(shape.dimensions() for shape in input_layouts)
  output_abstract_arrays = abs_eval_fn(*tuple(ShapedArray(shape.dimensions(), shape.element_type())
                                              for shape in input_layouts[:len(inputs)]),
                                       **description)
  if isinstance(output_abstract_arrays, ShapedArray):
    output_abstract_arrays = (output_abstract_arrays,)
    assert not multiple_results
  else:
    assert multiple_results
  output_shapes = tuple(array.shape for array in output_abstract_arrays)
  output_dtypes = tuple(array.dtype for array in output_abstract_arrays)
  output_layouts = map(lambda shape: range(len(shape) - 1, -1, -1), output_shapes)
  target_name = _cpu_signature(func,
                               input_dtypes,
                               input_dimensions,
                               output_dtypes,
                               output_shapes,
                               multiple_results)
  output_layouts = [xla_client.Shape.array_shape(*arg)
                    for arg in zip(output_dtypes, output_shapes, output_layouts)]
  output_layouts = (xla_client.Shape.tuple_shape(output_layouts)
                    if multiple_results else
                    output_layouts[0])
  return target_name, tuple(inputs) + tuple(info_inputs), input_layouts, output_layouts
