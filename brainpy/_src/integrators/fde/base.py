# -*- coding: utf-8 -*-

from typing import Union, Callable, Dict

import jax.numpy as jnp

import brainpy.math as bm
from brainpy.errors import UnsupportedError
from brainpy._src.integrators.base import Integrator
from brainpy._src.integrators.utils import get_args
from brainpy.check import is_integer

__all__ = [
  'FDEIntegrator'
]


class FDEIntegrator(Integrator):
  """Numerical integrator for fractional differential equations (FEDs).

  Parameters
  ----------
  f : callable
    The derivative function.
  alpha: int, float, jnp.ndarray, bm.ndarray, sequence
    The fractional-order of the derivative function.
  dt: float, int
    The numerical precision.
  name: str
    The integrator name.
  """

  alpha: bm.Array
  """The fraction order for each variable."""

  dt: Union[float, int]
  """The numerical integration precision."""

  f: Callable
  """The fraction derivative function."""

  def __init__(
      self,
      f: Callable,
      alpha,
      num_memory: int,
      dt: float = None,
      name: str = None,
      state_delays: Dict[str, Union[bm.LengthDelay, bm.TimeDelay]] = None,
  ):
    dt = bm.get_dt() if dt is None else dt
    parses = get_args(f)
    variables = parses[0]  # variable names, (before 't')
    parameters = parses[1]  # parameter names, (after 't')
    arguments = parses[2]  # function arguments

    # memory length
    is_integer(num_memory, 'num_memory', allow_none=False, min_bound=1)
    self.num_memory = num_memory

    # super initialization
    super(FDEIntegrator, self).__init__(name=name,
                                        variables=variables,
                                        parameters=parameters,
                                        arguments=arguments,
                                        dt=dt,
                                        state_delays=state_delays)

    # derivative function
    self.f = f

    # fractional-order
    if isinstance(alpha, (int, float)):
      alpha = jnp.ones(len(self.variables)) * alpha
    elif isinstance(alpha, (jnp.ndarray, bm.ndarray)):
      alpha = bm.as_jax(alpha)
    elif isinstance(alpha, (list, tuple)):
      for a in alpha:
        assert isinstance(a, (float, int)), (f'Must be a tuple/list of int/float, '
                                             f'but we got {type(a)}: {a}')
      alpha = bm.as_jax(alpha)
    else:
      raise UnsupportedError(f'Do not support {type(alpha)}, please '
                             f'set fractional-order as number/tuple/list/tensor.')
    if len(alpha) != len(self.variables):
      raise ValueError(f'There are {len(self.variables)} variables, '
                       f'while we only got {len(alpha)} fractional-order '
                       f'settings: {alpha}')
    self.alpha = alpha

