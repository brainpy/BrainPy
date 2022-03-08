# -*- coding: utf-8 -*-

from typing import Union, Sequence, Dict, Any, Callable

import jax.numpy as jnp
import numpy as onp

import brainpy.math as bm
from brainpy.initialize import Initializer
from brainpy.types import Tensor, Shape

__all__ = [
  'tensor_sum',
  'init_param',
]


def tensor_sum(values: Union[Sequence[Tensor], Dict[Any, Tensor], Tensor]):
  if isinstance(values, (bm.ndarray, jnp.ndarray)):
    return values
  if isinstance(values, dict):
    values = list(values.values())
  elif isinstance(values, (tuple, list)):
    values = list(values)
  else:
    raise ValueError('Unknown types of tensors.')
  res = values[0]
  for v in values[1:]:
    res = res + v
  return res


def init_param(param: Union[Callable, Initializer, bm.ndarray, jnp.ndarray],
               size: Shape):
  """Initialize parameters.

  Parameters
  ----------
  param: callable, Initializer, bm.ndarray, jnp.ndarray
    The initialization of the parameter.
    - If it is None, the created parameter will be None.
    - If it is a callable function :math:`f`, the ``f(size)`` will be returned.
    - If it is an instance of :py:class:`brainpy.init.Initializer``, the ``f(size)`` will be returned.
    - If it is a tensor, then this function check whether ``tensor.shape`` is equal to the given ``size``.
  size: int, sequence of int
    The shape of the parameter.
  """
  if param is None:
    return None
  elif callable(param):
    param = param(size)
  elif isinstance(param, (onp.ndarray, jnp.ndarray)):
    param = bm.asarray(param)
  elif isinstance(param, (bm.JaxArray,)):
    param = param
  else:
    raise ValueError(f'Unknown param type {type(param)}: {param}')
  assert param.shape == size, f'"param.shape" is not the required size {size}'
  return param

