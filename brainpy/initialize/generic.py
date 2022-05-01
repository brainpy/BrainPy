# -*- coding: utf-8 -*-

from typing import Union, Callable

import jax.numpy as jnp
import numpy as onp

import brainpy.math as bm
from brainpy.tools.others import to_size
from brainpy.types import Shape
from .base import Initializer

__all__ = [
  'init_param',
]


def init_param(
    param: Union[Callable, Initializer, bm.ndarray, jnp.ndarray, float, int, bool],
    size: Shape,
    allow_none: bool = True,
):
  """Initialize parameters.

  Parameters
  ----------
  param: callable, Initializer, bm.ndarray, jnp.ndarray, float, int, bool
    The initialization of the parameter.
    - If it is None, the created parameter will be None.
    - If it is a callable function :math:`f`, the ``f(size)`` will be returned.
    - If it is an instance of :py:class:`brainpy.init.Initializer``, the ``f(size)`` will be returned.
    - If it is a tensor, then this function check whether ``tensor.shape`` is equal to the given ``size``.
  size: int, sequence of int
    The shape of the parameter.
  allow_none: bool
    Whether allow the parameter is None.

  Returns
  -------
  param: JaxArray, float, None
    The initialized parameter.
  """
  size = to_size(size)
  if param is None:
    if allow_none:
      return None
    else:
      raise ValueError(f'Expect a parameter with type of float, JaxArray, Initializer, or '
                       f'Callable function, but we got None. ')
  elif isinstance(param, (float, int, bool)):
    return param
  elif callable(param):
    param = bm.asarray(param(size))
  elif isinstance(param, (onp.ndarray, jnp.ndarray)):
    param = bm.asarray(param)
  elif isinstance(param, bm.Variable):
    param = param
  elif isinstance(param, bm.JaxArray):
    param = param
  else:
    raise ValueError(f'Unknown param type {type(param)}: {param}')
  if param.shape != () and param.shape != (1,) and param.shape != size:
    raise ValueError(f'The shape of the parameters should be (), (1,) '
                     f'or {size}, but we got {param.shape}')
  return param
