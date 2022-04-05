# -*- coding: utf-8 -*-

import warnings
from typing import Union, Callable

import jax.numpy as jnp

from brainpy.math.delayvars import TimeDelay


__all__ = [
  'FixedLenDelay'
]


def FixedLenDelay(shape,
                  delay_len: Union[float, int],
                  before_t0: Union[Callable, jnp.ndarray, float, int] = None,
                  t0: Union[float, int] = 0.,
                  dt: Union[float, int] = None,
                  name: str = None,
                  interp_method='linear_interp', ):
  """Delay variable which has a fixed delay length.

  .. deprecated:: 2.1.2
     Please use "brainpy.math.TimeDelay" instead.

  See Also
  --------
  TimeDelay

  """
  warnings.warn('Please use "brainpy.math.TimeDelay" instead. '
                '"brainpy.math.FixedLenDelay" is deprecated since version 2.1.2. ',
                DeprecationWarning)
  return TimeDelay(inits=jnp.zeros(shape),
                   delay_len=delay_len,
                   before_t0=before_t0,
                   t0=t0,
                   dt=dt,
                   name=name,
                   interp_method=interp_method)

