# -*- coding: utf-8 -*-

import collections.abc
import ctypes
from functools import partial
from types import LambdaType
from typing import Callable, Union, Sequence

import jax.numpy as jnp
import numba
import numpy as np
from jax import core
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla
from numba.core.dispatcher import Dispatcher

from cpu import _func_cpu_translation
from gpu import _func_gpu_translation

_lambda_no = 0

def register_op(
    func: Callable,
    out_shapes: Union[Callable, ShapedArray, Sequence[ShapedArray]]
):
  """
  Converting the numba-jitted function in a Jax/XLA compatible primitive.
  Parameters
  ----------
  func:
  out_shapes

  Returns
  -------

  """
  # primitive
  prim = core.Primitive(f'_lambda_func{_lambda_no}'
                        if (isinstance(func, LambdaType) and func.__name__ == "<lambda>")
                        else func.__name__)
  prim.multiple_results = True

  # user defined function
  if not isinstance(func, Dispatcher):
    func = numba.jit(fastmath=True, nopython=True)(func)

  # output shape evaluation function
  def abs_eval_rule(*input_shapes):
    if callable(out_shapes):
      shapes = out_shapes(*input_shapes)
    elif isinstance(out_shapes, ShapedArray):
      shapes = [out_shapes]
    elif isinstance(out_shapes, (tuple, list)):
      shapes = out_shapes
      for elem in out_shapes:
        if not isinstance(elem, ShapedArray):
          raise ValueError(f'Elements in "out_shapes" must be instances of '
                           f'jax.abstract_arrays.ShapedArray, but we got '
                           f'{type(elem)}: {elem}')
    else:
      raise ValueError(f'Unknown type {type(out_shapes)}, only '
                       f'supports function, ShapedArray or '
                       f'list/tuple of ShapedArray.')

    # output shapes
    if not isinstance(shapes, collections.abc.Collection):
      return [shapes]
    else:
      return shapes

  # output evaluation function
  def eval_rule(*inputs):
    # compute the output shapes
    output_shapes = abs_eval_rule(*inputs)
    # Preallocate the outputs
    outputs = tuple(np.zeros(shape.shape, dtype=shape.dtype) for shape in output_shapes)
    # convert inputs to a tuple
    inputs = tuple(np.asarray(arg) for arg in inputs)
    # call the kernel
    func(outputs, inputs)
    # Return the outputs
    return tuple(outputs)

  def bind_primitive(*inputs):
    result = prim.bind(*inputs)
    return result[0] if len(result) == 1 else result

  # binding
  prim.def_abstract_eval(abs_eval_rule)
  prim.def_impl(eval_rule)
  # registering
  xla.backend_specific_translations['cpu'][prim] = partial(_func_cpu_translation, func, abs_eval_rule)
  xla.backend_specific_translations['gpu'][prim] = partial(_func_gpu_translation, func, abs_eval_rule)

  return bind_primitive


if __name__ == '__main__':
  def abs_eval(*ins):
    return ins

  import brainpy as bp
  bp.math.set_platform('cpu')

  def custom_op(outs, ins):
    y, y1 = outs
    x, x2 = ins
    y[:] = x + 1
    y1[:] = x2 + 2


  z = jnp.ones((1, 2), dtype=jnp.float32)
  op = register_op(custom_op, abs_eval)

  from jax import jit
  jit_op = jit(op)

  print(jit_op(z, z))
