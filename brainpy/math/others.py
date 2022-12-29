# -*- coding: utf-8 -*-


from typing import Optional

import jax.numpy as jnp

from brainpy import check, tools
from .environment import get_dt, get_int

__all__ = [
  'shared_args_over_time'
]


def shared_args_over_time(num_step: Optional[int] = None,
                          duration: Optional[float] = None,
                          dt: Optional[float] = None,
                          t0: float = 0.,
                          include_dt: bool = True):
  """Form a shared argument over time for the inference of a :py:class:`~.DynamicalSystem`.

  Parameters
  ----------
  num_step: int
    The number of time step. Provide either ``duration`` or ``num_step``.
  duration: float
    The total duration. Provide either ``duration`` or ``num_step``.
  dt: float
    The duration for each time step.
  t0: float
    The start time.
  include_dt: bool
    Produce the time steps at every time step.

  Returns
  -------
  shared: DotDict
    The shared arguments over the given time.
  """
  dt = get_dt() if dt is None else dt
  check.is_float(dt, 'dt', allow_none=False)
  if duration is None:
    check.is_integer(num_step, 'num_step', allow_none=False)
  else:
    check.is_float(duration, 'duration', allow_none=False)
    num_step = int(duration / dt)
  r = tools.DotDict(i=jnp.arange(num_step, dtype=get_int()))
  r['t'] = r['i'] * dt + t0
  if include_dt:
    r['dt'] = jnp.ones_like(r['t']) * dt
  return r
