# -*- coding: utf-8 -*-

import ctypes
from functools import partial

from jax.interpreters import xla, mlir
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call
from numba import types, carray, cfunc

from .utils import _shape_to_layout


__all__ = [
  'register_numba_xla_cpu_translation_rule',
  'register_numba_mlir_cpu_translation_rule',
]

ctypes.pythonapi.PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object


def _cpu_signature(
    kernel,
    input_dtypes,
    input_shapes,
    output_dtypes,
    output_shapes,
    debug: bool = False
):
  code_scope = dict(
    func_to_call=kernel,
    input_shapes=input_shapes,
    input_dtypes=input_dtypes,
    output_shapes=output_shapes,
    output_dtypes=output_dtypes,
    carray=carray,
  )

  # inputs, outputs, arguments
  args_in = [f'in{i} = carray(input_ptrs[{i}], input_shapes[{i}], dtype=input_dtypes[{i}])'
             for i in range(len(input_shapes))]
  args_out = [f'out{i} = carray(output_ptrs[{i}], output_shapes[{i}], dtype=output_dtypes[{i}])'
              for i in range(len(output_shapes))]
  args_call = [f'in{i}' for i in range(len(input_shapes))] + [f'out{i}' for i in range(len(output_shapes))]

  # function body
  code_string = '''
  def xla_cpu_custom_call_target(output_ptrs, input_ptrs):
    {args_in}
    {args_out}
    func_to_call({args_call})
    '''.format(args_in="\n    ".join(args_in),
               args_out="\n    ".join(args_out),
               args_call=", ".join(args_call))
  if debug: print(code_string)
  exec(compile(code_string.strip(), '', 'exec'), code_scope)

  # register
  new_f = code_scope['xla_cpu_custom_call_target']
  xla_c_rule = cfunc(types.void(types.CPointer(types.voidptr), types.CPointer(types.voidptr)))(new_f)
  target_name = f'numba_custom_call_{str(xla_c_rule.address)}'
  capsule = ctypes.pythonapi.PyCapsule_New(xla_c_rule.address, b"xla._CUSTOM_CALL_TARGET", None)
  xla_client.register_custom_call_target(target_name, capsule, "cpu")

  return target_name


def _numba_xla_cpu_translation_rule(kernel, debug: bool, c, *ins, **kwargs):
  outs = kwargs['outs']

  # output information
  output_shapes = tuple(out.shape for out in outs)
  output_dtypes = tuple(out.dtype for out in outs)
  output_layouts = map(lambda shape: range(len(shape) - 1, -1, -1), output_shapes)
  output_infos = [xla_client.Shape.array_shape(*arg) for arg in zip(output_dtypes, output_shapes, output_layouts)]
  output_infos = xla_client.Shape.tuple_shape(output_infos)

  # input information
  input_layouts = tuple(c.get_shape(arg) for arg in ins)
  input_dtypes = tuple(inp.element_type() for inp in input_layouts)
  input_shapes = tuple(inp.dimensions() for inp in input_layouts)

  # compiling
  target_name = _cpu_signature(kernel,
                               input_dtypes,
                               input_shapes,
                               output_dtypes,
                               output_shapes,
                               debug=debug)

  # call
  return xla_client.ops.CustomCallWithLayout(
    c,
    target_name.encode("ascii"),
    operands=tuple(ins),
    operand_shapes_with_layout=input_layouts,
    shape_with_layout=output_infos,
  )


def register_numba_xla_cpu_translation_rule(primitive, cpu_kernel, debug=False):
  xla.backend_specific_translations['cpu'][primitive] = partial(_numba_xla_cpu_translation_rule,
                                                                cpu_kernel,
                                                                debug)


def _numba_mlir_cpu_translation_rule(kernel, debug: bool, ctx, *ins, **kwargs):
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
  code_scope = dict(func_to_call=kernel, input_shapes=input_shapes, input_dtypes=input_dtypes,
                    output_shapes=output_shapes, output_dtypes=output_dtypes, carray=carray)
  args_in = [f'in{i} = carray(input_ptrs[{i}], input_shapes[{i}], dtype=input_dtypes[{i}])'
             for i in range(len(input_shapes))]
  args_out = [f'out{i} = carray(output_ptrs[{i}], output_shapes[{i}], dtype=output_dtypes[{i}])'
              for i in range(len(output_shapes))]
  args_call = [f'in{i}' for i in range(len(input_shapes))] + [f'out{i}' for i in range(len(output_shapes))]
  code_string = '''
  def numba_cpu_custom_call_target(output_ptrs, input_ptrs):
    {args_in}
    {args_out}
    func_to_call({args_call})
      '''.format(args_in="\n    ".join(args_in),
                 args_out="\n    ".join(args_out),
                 args_call=", ".join(args_call))
  if debug: print(code_string)
  exec(compile(code_string.strip(), '', 'exec'), code_scope)
  new_f = code_scope['numba_cpu_custom_call_target']

  # register
  xla_c_rule = cfunc(types.void(types.CPointer(types.voidptr), types.CPointer(types.voidptr)))(new_f)
  target_name = f'numba_custom_call_{str(xla_c_rule.address)}'
  capsule = ctypes.pythonapi.PyCapsule_New(xla_c_rule.address, b"xla._CUSTOM_CALL_TARGET", None)
  xla_client.register_custom_call_target(target_name, capsule, "cpu")

  # call
  call = custom_call(call_target_name=target_name,
                     operands=list(ins),
                     operand_layouts=list(input_layouts),
                     result_layouts=list(output_layouts),
                     result_types=list(result_types)).results
  return call


def register_numba_mlir_cpu_translation_rule(primitive, cpu_kernel, debug=False):
  rule = partial(_numba_mlir_cpu_translation_rule, cpu_kernel, debug)
  mlir.register_lowering(primitive, rule, platform='cpu')


