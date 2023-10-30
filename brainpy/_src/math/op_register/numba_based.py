# -*- coding: utf-8 -*-

import ctypes
from functools import partial

from jax.interpreters import xla
from jax.lib import xla_client
from numba import types, carray, cfunc

__all__ = [
  'register_numba_cpu_translation_rule',
]

ctypes.pythonapi.PyCapsule_New.argtypes = [
  ctypes.c_void_p,  # void* pointer
  ctypes.c_char_p,  # const char *name
  ctypes.c_void_p,  # PyCapsule_Destructor destructor
]
ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object


def _cpu_signature(
    kernel,
    input_dtypes,
    input_shapes,
    output_dtypes,
    output_shapes,
    debug: bool = False
):
  # kernel_key = str(id(kernel))
  # input_keys = [f'{dtype}({shape})' for dtype, shape in zip(input_dtypes, input_shapes)]
  # output_keys = [f'{dtype}({shape})' for dtype, shape in zip(output_dtypes, output_shapes)]
  # key = f'{kernel_key}-ins=[{", ".join(input_keys)}]-outs=[{", ".join(output_keys)}]'
  # if key not in __cache:

  code_scope = dict(
    func_to_call=kernel,
    input_shapes=input_shapes,
    input_dtypes=input_dtypes,
    output_shapes=output_shapes,
    output_dtypes=output_dtypes,
    carray=carray,
  )

  # inputs
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
    '''.format(args_in="\n  ".join(args_in),
               args_out="\n  ".join(args_out),
               args_call=", ".join(args_call))
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

  # else:
  #   target_name = __cache[key]
  return target_name


def _numba_cpu_translation_rule(prim, kernel, debug: bool, c, *ins):
  outs = prim.abstract_eval()[0]

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
    target_name,
    operands=tuple(ins),
    operand_shapes_with_layout=input_layouts,
    shape_with_layout=output_infos,
  )


def register_numba_cpu_translation_rule(primitive, cpu_kernel, debug=False):
  xla.backend_specific_translations['cpu'][primitive] = partial(_numba_cpu_translation_rule,
                                                                primitive, cpu_kernel, debug)
