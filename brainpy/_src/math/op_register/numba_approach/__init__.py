# -*- coding: utf-8 -*-
import ctypes
from functools import partial
from typing import Callable
from typing import Union, Sequence

import jax
from jax.interpreters import xla, batching, ad, mlir
from jax.lib import xla_client
from jax.tree_util import tree_map
from jaxlib.hlo_helpers import custom_call

from brainpy._src.dependency_check import import_numba
from brainpy._src.math.ndarray import Array
from brainpy._src.math.object_transform.base import BrainPyObject
from brainpy._src.math.op_register.utils import _shape_to_layout
from brainpy.errors import PackageMissingError
from .cpu_translation import _cpu_translation, compile_cpu_signature_with_numba

numba = import_numba(error_if_not_found=False)
if numba is not None:
  from numba import types, carray, cfunc

__all__ = [
  'CustomOpByNumba',
  'register_op_with_numba_xla',
  'compile_cpu_signature_with_numba',
]


def _transform_to_shapedarray(a):
  return jax.core.ShapedArray(a.shape, a.dtype)


def convert_shapedarray_to_shapedtypestruct(shaped_array):
  return jax.ShapeDtypeStruct(shape=shaped_array.shape, dtype=shaped_array.dtype)


class CustomOpByNumba(BrainPyObject):
  """Creating a XLA custom call operator with Numba JIT on CPU backend.

  Parameters
  ----------
  name: str
    The name of operator.
  eval_shape: callable
    The function to evaluate the shape and dtype of the output according to the input.
    This function should receive the abstract information of inputs, and return the
    abstract information of the outputs. For example:

    >>> def eval_shape(inp1_info, inp2_info, inp3_info, ...):
    >>>   return out1_info, out2_info
  con_compute: callable
    The function to make the concrete computation. This function receives inputs,
    and returns outputs. For example:

    >>> def con_compute(inp1, inp2, inp3, ..., out1, out2, ...):
    >>>   pass
  """

  def __init__(
      self,
      eval_shape: Callable = None,
      con_compute: Callable = None,
      name: str = None,
      batching_translation: Callable = None,
      jvp_translation: Callable = None,
      transpose_translation: Callable = None,
      multiple_results: bool = True,
  ):
    super().__init__(name=name)

    # abstract evaluation function
    if eval_shape is None:
      raise ValueError('Must provide "eval_shape" for abstract evaluation.')
    self.eval_shape = eval_shape

    # cpu function
    cpu_func = con_compute

    # register OP
    if jax.__version__ > '0.4.23':
      self.op_method = 'mlir'
      self.op = register_op_with_numba_mlir(
        self.name,
        cpu_func=cpu_func,
        out_shapes=eval_shape,
        gpu_func_translation=None,
        batching_translation=batching_translation,
        jvp_translation=jvp_translation,
        transpose_translation=transpose_translation,
        multiple_results=multiple_results,
      )
    else:
      self.op_method = 'xla'
      self.op = register_op_with_numba_xla(
        self.name,
        cpu_func=cpu_func,
        out_shapes=eval_shape,
        batching_translation=batching_translation,
        jvp_translation=jvp_translation,
        transpose_translation=transpose_translation,
        multiple_results=multiple_results,
      )

  def __call__(self, *args, **kwargs):
    args = tree_map(lambda a: a.value if isinstance(a, Array) else a,
                    args, is_leaf=lambda a: isinstance(a, Array))
    kwargs = tree_map(lambda a: a.value if isinstance(a, Array) else a,
                      kwargs, is_leaf=lambda a: isinstance(a, Array))
    res = self.op.bind(*args, **kwargs)
    return res


def register_op_with_numba_xla(
    op_name: str,
    cpu_func: Callable,
    out_shapes: Union[Callable, jax.core.ShapedArray, Sequence[jax.core.ShapedArray]],
    gpu_func_translation: Callable = None,
    batching_translation: Callable = None,
    jvp_translation: Callable = None,
    transpose_translation: Callable = None,
    multiple_results: bool = False,
):
  """
  Converting the numba-jitted function in a Jax/XLA compatible primitive.

  Parameters
  ----------
  op_name: str
    Name of the operators.

  cpu_func: Callable
    A callable numba-jitted function or pure function (can be lambda function) running on CPU.

  out_shapes: Callable, ShapedArray, Sequence[ShapedArray], default = None
    Outputs shapes of target function. `out_shapes` can be a `ShapedArray` or
    a sequence of `ShapedArray`. If it is a function, it takes as input the argument
    shapes and dtypes and should return correct output shapes of `ShapedArray`.

  gpu_func_translation: Callable
    A callable cuda-jitted kernel running on GPU.

  batching_translation: Callable
    The batching translation for the primitive.

  jvp_translation: Callable
    The forward autodiff translation rule.

  transpose_translation: Callable
    The backward autodiff translation rule.

  multiple_results: bool
    Whether the primitive returns multiple results. Default is False.

  Returns
  -------
  op: core.Primitive
    A JAX Primitive object.
  """

  if numba is None:
    raise PackageMissingError.by_purpose('numba', 'custom op with numba')

  if out_shapes is None:
    raise RuntimeError('out_shapes cannot be None. It can be a `ShapedArray` or '
                       'a sequence of `ShapedArray`. If it is a function, it takes as input the argument '
                       'shapes and dtypes and should return correct output shapes of `ShapedArray`.')

  prim = jax.core.Primitive(op_name)
  prim.multiple_results = multiple_results

  # user defined function
  from numba.core.dispatcher import Dispatcher
  if not isinstance(cpu_func, Dispatcher):
    cpu_func = numba.jit(fastmath=True, nopython=True)(cpu_func)

  # output shape evaluation function
  def abs_eval_rule(*input_shapes, **info):
    if callable(out_shapes):
      shapes = out_shapes(*input_shapes, **info)
    else:
      shapes = out_shapes

    if isinstance(shapes, jax.core.ShapedArray):
      assert not multiple_results, "multiple_results is True, while the abstract evaluation returns only one data."
    elif isinstance(shapes, (tuple, list)):
      assert multiple_results, "multiple_results is False, while the abstract evaluation returns multiple data."
      for elem in shapes:
        if not isinstance(elem, jax.core.ShapedArray):
          raise ValueError(f'Elements in "out_shapes" must be instances of '
                           f'jax.abstract_arrays.ShapedArray, but we got '
                           f'{type(elem)}: {elem}')
    else:
      raise ValueError(f'Unknown type {type(shapes)}, only '
                       f'supports function, ShapedArray or '
                       f'list/tuple of ShapedArray.')
    return shapes

  # cpu function
  prim.def_abstract_eval(abs_eval_rule)
  prim.def_impl(partial(xla.apply_primitive, prim))
  xla.backend_specific_translations['cpu'][prim] = partial(_cpu_translation,
                                                           cpu_func,
                                                           abs_eval_rule,
                                                           multiple_results)

  # gpu function
  if gpu_func_translation is not None:
    xla.backend_specific_translations['gpu'][prim] = gpu_func_translation

  # batching
  if batching_translation is not None:
    batching.primitive_batchers[prim] = batching_translation

  # jvp
  if jvp_translation is not None:
    ad.primitive_jvps[prim] = jvp_translation

  # transpose
  if transpose_translation is not None:
    ad.primitive_transposes[prim] = transpose_translation

  return prim


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
  if len(output_shapes) > 1:
    args_out = [f'out{i} = carray(output_ptrs[{i}], output_shapes[{i}], dtype=output_dtypes[{i}])'
                for i in range(len(output_shapes))]
    sig = types.void(types.CPointer(types.voidptr), types.CPointer(types.voidptr))
  else:
    args_out = [f'out0 = carray(output_ptrs, output_shapes[0], dtype=output_dtypes[0])']
    sig = types.void(types.voidptr, types.CPointer(types.voidptr))
  args_call = [f'out{i}' for i in range(len(output_shapes))] + [f'in{i}' for i in range(len(input_shapes))]
  code_string = '''
  def numba_cpu_custom_call_target(output_ptrs, input_ptrs):
      {args_out}
      {args_in}
      func_to_call({args_call})
      '''.format(args_out="\n    ".join(args_out), args_in="\n    ".join(args_in), args_call=", ".join(args_call))

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


def register_op_with_numba_mlir(
    op_name: str,
    cpu_func: Callable,
    out_shapes: Union[Callable, jax.core.ShapedArray, Sequence[jax.core.ShapedArray]],
    gpu_func_translation: Callable = None,
    batching_translation: Callable = None,
    jvp_translation: Callable = None,
    transpose_translation: Callable = None,
    multiple_results: bool = False,
):
  if numba is None:
    raise PackageMissingError.by_purpose('numba', 'custom op with numba')

  if out_shapes is None:
    raise RuntimeError('out_shapes cannot be None. It can be a `ShapedArray` or '
                       'a sequence of `ShapedArray`. If it is a function, it takes as input the argument '
                       'shapes and dtypes and should return correct output shapes of `ShapedArray`.')

  prim = jax.core.Primitive(op_name)
  prim.multiple_results = multiple_results

  from numba.core.dispatcher import Dispatcher
  if not isinstance(cpu_func, Dispatcher):
    cpu_func = numba.jit(fastmath=True, nopython=True)(cpu_func)

  def abs_eval_rule(*input_shapes, **info):
    if callable(out_shapes):
      shapes = out_shapes(*input_shapes, **info)
    else:
      shapes = out_shapes

    if isinstance(shapes, jax.core.ShapedArray):
      assert not multiple_results, "multiple_results is True, while the abstract evaluation returns only one data."
    elif isinstance(shapes, (tuple, list)):
      assert multiple_results, "multiple_results is False, while the abstract evaluation returns multiple data."
      for elem in shapes:
        if not isinstance(elem, jax.core.ShapedArray):
          raise ValueError(f'Elements in "out_shapes" must be instances of '
                           f'jax.abstract_arrays.ShapedArray, but we got '
                           f'{type(elem)}: {elem}')
    else:
      raise ValueError(f'Unknown type {type(shapes)}, only '
                       f'supports function, ShapedArray or '
                       f'list/tuple of ShapedArray.')
    return shapes

  prim.def_abstract_eval(abs_eval_rule)
  prim.def_impl(partial(xla.apply_primitive, prim))

  def cpu_translation_rule(ctx, *ins, **kwargs):
    return _numba_mlir_cpu_translation_rule(cpu_func, False, ctx, *ins, **kwargs)

  mlir.register_lowering(prim, cpu_translation_rule, platform='cpu')

  if gpu_func_translation is not None:
    mlir.register_lowering(prim, gpu_func_translation, platform='gpu')

  if batching_translation is not None:
    jax.interpreters.batching.primitive_batchers[prim] = batching_translation

  if jvp_translation is not None:
    jax.interpreters.ad.primitive_jvps[prim] = jvp_translation

  if transpose_translation is not None:
    jax.interpreters.ad.primitive_transposes[prim] = transpose_translation

  return prim
