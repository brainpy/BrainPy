# -*- coding: utf-8 -*-

from typing import Union, Sequence, Callable

from jax.core import ShapedArray
from jax.tree_util import tree_map

from brainpy.base import Base
from brainpy.math.jaxarray import JaxArray
from .utils import _check_brainpylib

try:
  import brainpylib
except ModuleNotFoundError:
  brainpylib = None

__all__ = [
  'XLACustomOp',
  'register_op',
]


class XLACustomOp(Base):
  """Creating a XLA custom call operator.

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

    >>> def con_compute(inp1, inp2, inp3, ...):
    >>>   return out1, out2
  cpu_func: callable
    The function defines the computation on CPU backend. Same as ``con_compute``.
  gpu_func: callable
    The function defines the computation on GPU backend. Currently, this function is not supportted.
  apply_cpu_func_to_gpu: bool
    Whether allows to apply CPU function on GPU backend. If True, the GPU data will move to CPU,
    and after calculation, the returned outputs on CPU backend will move to GPU.
  """

  def __init__(
      self,
      eval_shape: Callable = None,
      con_compute: Callable = None,
      cpu_func: Callable = None,
      gpu_func: Callable = None,
      apply_cpu_func_to_gpu: bool = False,
      name: str = None,
      batching_translation: Callable = None,
      jvp_translation: Callable = None,
      transpose_translation: Callable = None,
      multiple_results: bool = False,
  ):
    _check_brainpylib(register_op.__name__)
    super(XLACustomOp, self).__init__(name=name)

    # abstract evaluation function
    if eval_shape is None:
      raise ValueError('Must provide "eval_shape" for abstract evaluation.')

    # cpu function
    if con_compute is None:
      if cpu_func is None:
        raise ValueError('Must provide one of "cpu_func" or "con_compute".')
    else:
      cpu_func = con_compute

    # gpu function
    if gpu_func is None:
      gpu_func = None

    # register OP
    self.op = brainpylib.register_op_with_numba(
      self.name,
      cpu_func=cpu_func,
      gpu_func_translation=gpu_func,
      out_shapes=eval_shape,
      apply_cpu_func_to_gpu=apply_cpu_func_to_gpu,
      batching_translation=batching_translation,
      jvp_translation=jvp_translation,
      transpose_translation=transpose_translation,
      multiple_results=multiple_results,
    )

  def __call__(self, *args, **kwargs):
    args = tree_map(lambda a: a.value if isinstance(a, JaxArray) else a,
                    args, is_leaf=lambda a: isinstance(a, JaxArray))
    kwargs = tree_map(lambda a: a.value if isinstance(a, JaxArray) else a,
                      kwargs, is_leaf=lambda a: isinstance(a, JaxArray))
    res = self.op.bind(*args, **kwargs)
    return res


def register_op(
    name: str,
    eval_shape: Union[Callable, ShapedArray, Sequence[ShapedArray]],
    cpu_func: Callable,
    gpu_func: Callable = None,
    apply_cpu_func_to_gpu: bool = False
):
  """
  Converting the numba-jitted function in a Jax/XLA compatible primitive.

  Parameters
  ----------
  name: str
    Name of the operators.
  cpu_func: Callble
    A callable numba-jitted function or pure function (can be lambda function) running on CPU.
  gpu_func: Callable, default = None
    A callable cuda-jitted kernel running on GPU.
  eval_shape: Callable, ShapedArray, Sequence[ShapedArray], default = None
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
  f = brainpylib.register_op_with_numba(name,
                                        cpu_func=cpu_func,
                                        gpu_func_translation=gpu_func,
                                        out_shapes=eval_shape,
                                        apply_cpu_func_to_gpu=apply_cpu_func_to_gpu)

  def fixed_op(*inputs, **info):
    inputs = tuple([i.value if isinstance(i, JaxArray) else i for i in inputs])
    res = f.bind(*inputs, **info)
    return res

  return fixed_op
