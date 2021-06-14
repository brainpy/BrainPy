# -*- coding: utf-8 -*-

from . import ode
from . import sde

__all__ = [
  'SUPPORTED_ODE_METHODS',
  'SUPPORTED_SDE_METHODS',

  'odeint',
  'sdeint',
  'ddeint',
  'fdeint',

  'set_default_odeint', 'get_default_odeint',
  'set_default_sdeint', 'get_default_sdeint',
]

_DEFAULT_ODE_METHOD = 'euler'
_DEFAULT_SDE_METHOD = 'euler'
SUPPORTED_ODE_METHODS = [m for m in dir(ode) if not m.startswith('__') and callable(getattr(ode, m))]
SUPPORTED_SDE_METHODS = [m for m in dir(sde) if not m.startswith('__') and callable(getattr(sde, m))]


def _wrapper(f, module, method, **kwargs):
  integrator = getattr(module, method)
  return integrator(f, **kwargs)


def odeint(f=None, method=None, **kwargs):
  """Numerical integration for ODE.

  Parameters
  ----------
  f : callable
  method : str
  kwargs :

  Returns
  -------
  int_f : callable
      The numerical solver of `f`.
  """
  if method is None:
    method = _DEFAULT_ODE_METHOD
  if method not in SUPPORTED_ODE_METHODS:
    raise ValueError(f'Unknown ODE numerical method "{method}". Currently '
                     f'BrainPy only support: {SUPPORTED_ODE_METHODS}')

  if f is None:
    return lambda f: _wrapper(f, method=method, module=ode, **kwargs)
  else:
    return _wrapper(f, method=method, module=ode, **kwargs)


def sdeint(f=None, method=None, **kwargs):
  if method is None:
    method = _DEFAULT_SDE_METHOD
  if method not in SUPPORTED_SDE_METHODS:
    raise ValueError(f'Unknown SDE numerical method "{method}". Currently '
                     f'BrainPy only support: {SUPPORTED_SDE_METHODS}')

  if f is None:
    return lambda f: _wrapper(f, method=method, module=sde, **kwargs)
  else:
    return _wrapper(f, method=method, module=sde, **kwargs)


def ddeint():
  raise NotImplementedError


def fdeint():
  raise NotImplementedError


def set_default_odeint(method):
  """Set the default ODE numerical integrator method for differential equations.

  Parameters
  ----------
  method : str, callable
      Numerical integrator method.
  """
  if not isinstance(method, str):
    raise ValueError(f'Only support string, not {type(method)}.')
  if method not in SUPPORTED_ODE_METHODS:
    raise ValueError(f'Unsupported ODE numerical method: {method}.')

  global _DEFAULT_ODE_METHOD
  _DEFAULT_ODE_METHOD = method


def get_default_odeint():
  """Get the default ODE numerical integrator method.

  Returns
  -------
  method : str
      The default numerical integrator method.
  """
  return _DEFAULT_ODE_METHOD


def set_default_sdeint(method):
  """Set the default SDE numerical integrator method for differential equations.

  Parameters
  ----------
  method : str, callable
      Numerical integrator method.
  """
  if not isinstance(method, str):
    raise ValueError(f'Only support string, not {type(method)}.')
  if method not in SUPPORTED_SDE_METHODS:
    raise ValueError(f'Unsupported SDE numerical method: {method}.')

  global _DEFAULT_SDE_METHOD
  _DEFAULT_SDE_METHOD = method


def get_default_sdeint():
  """Get the default ODE numerical integrator method.

  Returns
  -------
  method : str
      The default numerical integrator method.
  """
  return _DEFAULT_SDE_METHOD
