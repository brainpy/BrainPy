# -*- coding: utf-8 -*-

import warnings
from typing import Callable

import brainpylib
from jax.tree_util import tree_map

from ..object_transform.base_object import BrainPyObject
from ..ndarray import Array

__all__ = [
  'XLACustomOp',
]


class XLACustomOp(BrainPyObject):
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

    .. deprecated:: 2.2.4.1
       No longer supported.
  """

  def __init__(
      self,
      eval_shape: Callable = None,
      con_compute: Callable = None,
      cpu_func: Callable = None,
      gpu_func: Callable = None,
      apply_cpu_func_to_gpu: bool = None,
      name: str = None,
      batching_translation: Callable = None,
      jvp_translation: Callable = None,
      transpose_translation: Callable = None,
      multiple_results: bool = True,
  ):
    super(XLACustomOp, self).__init__(name=name)

    if apply_cpu_func_to_gpu is not None:
      warnings.warn('"apply_cpu_func_to_gpu" has been removed.', UserWarning)

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

