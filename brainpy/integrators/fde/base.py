# -*- coding: utf-8 -*-

from typing import Union, Callable, Dict

import jax.numpy as jnp

import brainpy.math as bm
from brainpy.errors import UnsupportedError
from brainpy.integrators.base import Integrator
from brainpy.integrators.utils import get_args

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

  """The fraction order for each variable."""
  alpha: jnp.ndarray

  """The numerical integration precision."""
  dt: Union[float, int]

  """The fraction derivative function."""
  f: Callable

  def __init__(
      self,
      f: Callable,
      alpha,
      dt: float = None,
      name: str = None,
      state_delays: Dict[str, bm.AbstractDelay] = None,
  ):
    dt = bm.get_dt() if dt is None else dt
    parses = get_args(f)
    variables = parses[0]  # variable names, (before 't')
    parameters = parses[1]  # parameter names, (after 't')
    arguments = parses[2]  # function arguments

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
      alpha = bm.as_device_array(alpha)
    elif isinstance(alpha, (list, tuple)):
      for a in alpha:
        assert isinstance(a, (float, int)), (f'Must be a tuple/list of int/float, '
                                             f'but we got {type(a)}: {a}')
      alpha = jnp.asarray(alpha)
    else:
      raise UnsupportedError(f'Do not support {type(alpha)}, please '
                             f'set fractional-order as number/tuple/list/tensor.')
    if len(alpha) != len(self.variables):
      raise ValueError(f'There are {len(self.variables)} variables, '
                       f'while we only got {len(alpha)} fractional-order '
                       f'settings: {alpha}')
    self.alpha = alpha

