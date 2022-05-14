# -*- coding: utf-8 -*-


import warnings
from typing import Union, Dict

import brainpy.math as bm
from .base import DDEIntegrator

__all__ = [
  'ddeint',
  'set_default_ddeint',
  'get_default_ddeint',
  'register_dde_integrator',
  'get_supported_methods',
]

name2method = {}

_DEFAULT_DDE_METHOD = 'euler'


def ddeint(
    f=None,
    method='euler',
    var_type: str = None,
    dt: Union[float, int] = None,
    name: str = None,
    show_code: bool = False,
    state_delays: Dict[str, bm.TimeDelay] = None,
    neutral_delays: Dict[str, bm.NeuTimeDelay] = None,
    **kwargs
):
  """Numerical integration for ODEs.

  .. deprecated:: 2.1.11
     Please use :py:func:`~.odeint` instead. This module will be removed since version 2.2.0.

  Parameters
  ----------
  f : callable, function
    The derivative function.
  method : str
    The shortcut name of the numerical integrator.
  var_type: str
    Variable type in the defined function.
  dt: float, int
    The time precision for integration.
  name: str
    The name.
  show_code: bool
    Whether show the formatted codes.
  state_delays: dict
    The state delay variables.
  neutral_delays: dict
    The neutral delay variable.

  Returns
  -------
  integral : DDEIntegrator
      The numerical solver of `f`.
  """
  warnings.warn('Please use "brainpy.dde.ddeint" instead. '
                '"brainpy.dde.ddeint" is deprecated since '
                'version 2.1.11. ', DeprecationWarning)

  method = _DEFAULT_DDE_METHOD if method is None else method
  if method not in name2method:
    raise ValueError(f'Unknown DDE numerical method "{method}". Currently '
                     f'BrainPy only support: {list(name2method.keys())}')

  if f is None:
    return lambda f: name2method[method](f,
                                         var_type=var_type,
                                         dt=dt,
                                         name=name,
                                         state_delays=state_delays,
                                         neutral_delays=neutral_delays,
                                         **kwargs)
  else:
    return name2method[method](f,
                               var_type=var_type,
                               dt=dt,
                               name=name,
                               state_delays=state_delays,
                               neutral_delays=neutral_delays,
                               **kwargs)


def set_default_ddeint(method):
  """Set the default ODE numerical integrator method for differential equations.

  Parameters
  ----------
  method : str, callable
      Numerical integrator method.
  """
  if not isinstance(method, str):
    raise ValueError(f'Only support string, not {type(method)}.')
  if method not in name2method:
    raise ValueError(f'Unsupported ODE_INT numerical method: {method}.')

  global _DEFAULT_DDE_METHOD
  _DEFAULT_DDE_METHOD = method


def get_default_ddeint():
  """Get the default ODE numerical integrator method.

  Returns
  -------
  method : str
      The default numerical integrator method.
  """
  return _DEFAULT_DDE_METHOD


def register_dde_integrator(name, integrator):
  """Register a new DDE integrator.

  Parameters
  ----------
  name: ste
  integrator: type
  """
  if name in name2method:
    raise ValueError(f'"{name}" has been registered in DDE integrators.')
  if not issubclass(integrator, DDEIntegrator):
    raise ValueError(f'"integrator" must be an instance of {DDEIntegrator.__name__}')
  name2method[name] = integrator


def get_supported_methods():
  """Get all supported numerical methods for DDEs."""
  return list(name2method.keys())
