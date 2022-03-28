# -*- coding: utf-8 -*-

import collections.abc
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
from numba import cuda

from cpu import _func_cpu_translation
from gpu import _func_gpu_translation

_lambda_no = 0


def register_op(
    cpu_func: Callable,
    op_name: str,
    out_shapes: Union[Callable, ShapedArray, Sequence[ShapedArray]],
    gpu_func: Callable = None,
    apply_cpu_func_to_gpu: bool = True,
):
  """
  Converting the numba-jitted function in a Jax/XLA compatible primitive.
  Parameters
  ----------
  cpu_func: Callble
    A callable numba-jitted function or pure function (can be lambda function) running on CPU.
  op_name: str
    Name of the operators.
  out_shapes: Callable, ShapedArray, Sequence[ShapedArray]
    Outputs shapes of target function. `out_shapes` can be a `jax.abstract_arrays.ShapedArray` or
    a sequence of `jax.abstract_arrays.ShapedArray`. If it is a function, it takes as input the argument
    shapes and dtypes and should return correct output shapes of `jax.abstract_arrays.ShapedArray`.
  gpu_func: Callable, default = None
    A callable cuda-jitted kernel running on GPU.
  apply_gpu_func_in_cpu: bool, default = True
    True when gpu_func is implemented on CPU and other logics(data transfer) is implemented on GPU.

  Returns
  -------
  A jitable JAX function.
  """
  if gpu_func is not None:
    raise RuntimeError('Currently cuda.jit function is not supported to convert into a Jax/XLA compatible primitive.' \
                     ' Please wait for future version to use gpu_func. Now we support to set apply_cpu_func_to_gpu = True' \
                     ' for a alternative method to run on GPU.')

  if (gpu_func is not None) and apply_cpu_func_to_gpu:
    raise RuntimeError("apply_cpu_func_to_gpu cannot be true if gpu_func is not None.")

  prim = core.Primitive(op_name)
  prim.multiple_results = True

  # user defined function
  if not isinstance(cpu_func, Dispatcher):
    cpu_func = numba.jit(fastmath=True, nopython=True)(cpu_func)

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
    cpu_func(outputs, inputs)
    # Return the outputs
    return tuple(outputs)

  def bind_primitive(*inputs):
    result = prim.bind(*inputs)
    return result[0] if len(result) == 1 else result

  # binding
  prim.def_abstract_eval(abs_eval_rule)
  prim.def_impl(eval_rule)
  # registering
  xla.backend_specific_translations['cpu'][prim] = partial(_func_cpu_translation, cpu_func, abs_eval_rule)
  if apply_cpu_func_to_gpu:
    xla.backend_specific_translations['gpu'][prim] = partial(_func_gpu_translation, cpu_func, abs_eval_rule)
    return bind_primitive

  if gpu_func is not None:
    if not isinstance(gpu_func, Dispatcher):
      gpu_func = cuda.jit(gpu_func)
    xla.backend_specific_translations['gpu'][prim] = partial(_func_gpu_translation, gpu_func, abs_eval_rule)

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
  op = register_op(cpu_func=custom_op, op_name='add', out_shapes=abs_eval, apply_cpu_func_to_gpu=True)

  from jax import jit

  jit_op = jit(op)

  print(jit_op(z, z))
