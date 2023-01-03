# -*- coding: utf-8 -*-

from typing import Dict, Union

import brainpy.math as bm
from .base import SDEIntegrator

__all__ = [
  'set_default_sdeint',
  'get_default_sdeint',
  'register_sde_integrator',
  'get_supported_methods',
]

name2method = {
}

_DEFAULT_SDE_METHOD = 'euler'


def sdeint(
    f=None,
    g=None,
    method=None,
    dt: float = None,
    name: str = None,
    show_code: bool = False,
    var_type: str = None,
    intg_type: str = None,
    wiener_type: str = None,
    state_delays: Dict[str, Union[bm.LengthDelay, bm.TimeDelay]] = None
):
  """Numerical integration for SDEs.

  Parameters
  ----------
  f : callable, function
    The derivative function.
  method : str
    The shortcut name of the numerical integrator.

  Returns
  -------
  integral : SDEIntegrator
      The numerical solver of `f`.
  """
  method = _DEFAULT_SDE_METHOD if method is None else method
  if method not in name2method:
    raise ValueError(f'Unknown SDE numerical method "{method}". Currently '
                     f'BrainPy only support: {list(name2method.keys())}')

  if f is not None and g is not None:
    return name2method[method](f=f,
                               g=g,
                               dt=dt,
                               name=name,
                               show_code=show_code,
                               var_type=var_type,
                               intg_type=intg_type,
                               wiener_type=wiener_type,
                               state_delays=state_delays)

  elif f is not None:
    return lambda g: name2method[method](f=f,
                                         g=g,
                                         dt=dt,
                                         name=name,
                                         show_code=show_code,
                                         var_type=var_type,
                                         intg_type=intg_type,
                                         wiener_type=wiener_type,
                                         state_delays=state_delays)

  elif g is not None:
    return lambda f: name2method[method](f=f,
                                         g=g,
                                         dt=dt,
                                         name=name,
                                         show_code=show_code,
                                         var_type=var_type,
                                         intg_type=intg_type,
                                         wiener_type=wiener_type,
                                         state_delays=state_delays)

  else:
    raise ValueError('Must provide "f" or "g".')


def set_default_sdeint(method):
  """Set the default SDE numerical integrator method for differential equations.

  Parameters
  ----------
  method : str, callable
      Numerical integrator method.
  """
  if not isinstance(method, str):
    raise ValueError(f'Only support string, not {type(method)}.')
  if method not in name2method:
    raise ValueError(f'Unsupported SDE_INT numerical method: {method}.')

  global _DEFAULT_SDE_METHOD
  _DEFAULT_SDE_METHOD = method


def get_default_sdeint():
  """Get the default SDE numerical integrator method.

  Returns
  -------
  method : str
      The default numerical integrator method.
  """
  return _DEFAULT_SDE_METHOD


def register_sde_integrator(name, integrator):
  """Register a new SDE integrator.

  Parameters
  ----------
  name: ste
  integrator: type
  """
  if name in name2method:
    raise ValueError(f'"{name}" has been registered in SDE integrators.')
  if not issubclass(integrator, SDEIntegrator):
    raise ValueError(f'"integrator" must be an instance of {SDEIntegrator.__name__}')
  name2method[name] = integrator


def get_supported_methods():
  """Get all supported numerical methods for DDEs."""
  return list(name2method.keys())
