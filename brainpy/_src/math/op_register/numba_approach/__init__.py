# -*- coding: utf-8 -*-

from functools import partial
from typing import Callable
from typing import Union, Sequence

import jax
from jax.interpreters import xla, batching, ad
from jax.tree_util import tree_map

from brainpy._src.dependency_check import import_numba
from brainpy._src.math.ndarray import Array
from brainpy._src.math.object_transform.base import BrainPyObject
from brainpy.errors import PackageMissingError
from .cpu_translation import _cpu_translation, compile_cpu_signature_with_numba

numba = import_numba(error_if_not_found=False)


__all__ = [
  'CustomOpByNumba',
  'register_op_with_numba',
  'compile_cpu_signature_with_numba',
]


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

    # cpu function
    cpu_func = con_compute

    # register OP
    self.op = register_op_with_numba(
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


def register_op_with_numba(
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

  if jax.__version__ > '0.4.23':
    raise RuntimeError(f'{CustomOpByNumba.__name__} and {register_op_with_numba.__name__} are '
                       f'only supported in JAX version <= 0.4.23. \n'
                       f'However, you can use brainpy.math.XLACustomOp to create a custom op with numba syntax. '
                       f'For more information, please refer to the documentation: '
                       f'https://brainpy.readthedocs.io/en/latest/tutorial_advanced/operator_custom_with_taichi.html.')

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
