# -*- coding: utf-8 -*-

import ctypes

from jax import dtypes, numpy as jnp
from jax.core import ShapedArray
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call
from jax.interpreters import mlir

from brainpy._src.dependency_check import import_numba
from brainpy._src.math.op_register.utils import _shape_to_layout

numba = import_numba(error_if_not_found=False)
ctypes.pythonapi.PyCapsule_New.argtypes = [
  ctypes.c_void_p,  # void* pointer
  ctypes.c_char_p,  # const char *name
  ctypes.c_void_p,  # PyCapsule_Destructor destructor
]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object

__all__ = [
  '_cpu_translation',
  'compile_cpu_signature_with_numba',
  '_numba_mlir_cpu_translation_rule',
]

if numba is not None:
  from numba import types, carray, cfunc


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
                               multiple_results,
                               debug=False)
  output_layouts = [xla_client.Shape.array_shape(*arg)
                    for arg in zip(output_dtypes, output_shapes, output_layouts)]
  output_layouts = (xla_client.Shape.tuple_shape(output_layouts)
                    if multiple_results else
                    output_layouts[0])
  return target_name, tuple(inputs) + tuple(info_inputs), input_layouts, output_layouts


def _numba_mlir_cpu_translation_rule(
    cpu_func,
    debug,
    ctx,
    *ins,
    **kwargs
):
  # output information
  outs = ctx.avals_out
  output_shapes = tuple([out.shape for out in outs])
  output_dtypes = tuple([out.dtype for out in outs])
  output_layouts = tuple([_shape_to_layout(out.shape) for out in outs])
  result_types = [mlir.aval_to_ir_type(out) for out in outs]

  # input information
  avals_in = ctx.avals_in
  input_layouts = [_shape_to_layout(a.shape) for a in avals_in]
  input_dtypes = tuple(inp.dtype for inp in avals_in)
  input_shapes = tuple(inp.shape for inp in avals_in)

  # compiling function
  code_scope = dict(func_to_call=cpu_func, input_shapes=input_shapes, input_dtypes=input_dtypes,
                    output_shapes=output_shapes, output_dtypes=output_dtypes, carray=carray)
  if len(input_shapes) > 1:
    args_in = [
      f'carray(input_ptrs[{i}], input_shapes[{i}], dtype=input_dtypes[{i}]),'
      for i in range(len(input_shapes))
    ]
    args_in = '(\n    ' + "\n    ".join(args_in) + '\n  )'
  else:
    args_in = 'carray(input_ptrs[0], input_shapes[0], dtype=input_dtypes[0])'
  if len(output_shapes) > 1:
    args_out = [
      f'carray(output_ptrs[{i}], output_shapes[{i}], dtype=output_dtypes[{i}]),'
      for i in range(len(output_shapes))
    ]
    args_out = '(\n    ' + "\n    ".join(args_out) + '\n  )'
    sig = types.void(types.CPointer(types.voidptr), types.CPointer(types.voidptr))
  else:
    args_out = 'carray(output_ptrs, output_shapes[0], dtype=output_dtypes[0])'
    sig = types.void(types.voidptr, types.CPointer(types.voidptr))
  # args_call = [f'out{i}' for i in range(len(output_shapes))] + [f'in{i}' for i in range(len(input_shapes))]
  code_string = '''
def numba_cpu_custom_call_target(output_ptrs, input_ptrs):
    args_out = {args_out}
    args_in = {args_in}
    func_to_call(args_out, args_in)
  '''.format(args_in=args_in,
             args_out=args_out)

  if debug:
    print(code_string)
  exec(compile(code_string.strip(), '', 'exec'), code_scope)
  new_f = code_scope['numba_cpu_custom_call_target']

  # register
  xla_c_rule = cfunc(sig)(new_f)
  target_name = f'numba_custom_call_{str(xla_c_rule.address)}'
  capsule = ctypes.pythonapi.PyCapsule_New(xla_c_rule.address, b"xla._CUSTOM_CALL_TARGET", None)
  xla_client.register_custom_call_target(target_name, capsule, "cpu")

  # call
  return custom_call(
    call_target_name=target_name,
    operands=ins,
    operand_layouts=list(input_layouts),
    result_layouts=list(output_layouts),
    result_types=list(result_types),
    has_side_effect=False,
  ).results
