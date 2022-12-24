# -*- coding: utf-8 -*-


import jax.numpy as jnp
import math

from brainpy import check, tools
from .environment import get_dt

__all__ = [
  'form_shared_args'
]


def form_shared_args(num_step: int = None,
                     duration: float = None,
                     dt: float = None,
                     t0: float = 0.,
                     include_dt: bool = True):
  """Form a shared argument for the inference of a :py:class:`~.DynamicalSystem`.

  Parameters
  ----------
  duration: float
  num_step: int
  dt: float
  t0: float
  include_dt: bool

  Returns
  -------
  shared: DotDict
    The shared arguments over the given time.
  """

  dt = get_dt() if dt is None else dt
  check.check_float(dt, 'dt', allow_none=False)
  if duration is None:
    check.check_integer(num_step, 'num_step', allow_none=False)
  else:
    check.check_float(duration, 'duration', allow_none=False)
    num_step = int(duration / dt)
  r = tools.DotDict(i=jnp.arange(num_step))
  r['t'] = r['i'] * dt + t0
  if include_dt:
    r['dt'] = jnp.ones_like(r['t']) * dt
  return r
