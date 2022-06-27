# -*- coding: utf-8 -*-

from typing import Union, Sequence, Callable

from jax.abstract_arrays import ShapedArray

from brainpy.math.jaxarray import JaxArray
from .utils import _check_brainpylib

try:
  import brainpylib
except ModuleNotFoundError:
  brainpylib = None

__all__ = [
  'register_op'
]


def register_op(
    op_name: str,
    cpu_func: Callable,
    out_shapes: Union[Callable, ShapedArray, Sequence[ShapedArray]],
    gpu_func: Callable = None,
    apply_cpu_func_to_gpu: bool = False
):
  """
  Converting the numba-jitted function in a Jax/XLA compatible primitive.

  Parameters
  ----------
  op_name: str
    Name of the operators.
  cpu_func: Callble
    A callable numba-jitted function or pure function (can be lambda function) running on CPU.
  gpu_func: Callable, default = None
    A callable cuda-jitted kernel running on GPU.
  out_shapes: Callable, ShapedArray, Sequence[ShapedArray], default = None
    Outputs shapes of target function. `out_shapes` can be a `ShapedArray` or
    a sequence of `ShapedArray`. If it is a function, it takes as input the argument
    shapes and dtypes and should return correct output shapes of `ShapedArray`.
  apply_cpu_func_to_gpu: bool, default = False
    True when gpu_func is implemented on CPU and other logics(data transfer) is implemented on GPU.

  Returns
  -------
  A jitable JAX function.
  """
  _check_brainpylib(register_op.__name__)
  f = brainpylib.register_op(op_name, cpu_func, gpu_func, out_shapes, apply_cpu_func_to_gpu)

  def fixed_op(*inputs):
    inputs = tuple([i.value if isinstance(i, JaxArray) else i for i in inputs])
    return f(*inputs)

  return fixed_op
