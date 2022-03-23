# -*- coding: utf-8 -*-

from .base import FDEIntegrator

__all__ = [
  'fdeint',
  'set_default_fdeint',
  'get_default_fdeint',
  'register_fde_integrator',
  'get_supported_methods',
]

name2method = {
}

_DEFAULT_DDE_METHOD = 'CaputoL1'


def fdeint(f=None, method='CaputoL1', **kwargs):
  """Numerical integration for FDEs.

  Parameters
  ----------
  f : callable, function
    The derivative function.
  method : str
    The shortcut name of the numerical integrator.

  Returns
  -------
  integral : FDEIntegrator
      The numerical solver of `f`.
  """
  method = _DEFAULT_DDE_METHOD if method is None else method
  if method not in name2method:
    raise ValueError(f'Unknown FDE numerical method "{method}". Currently '
                     f'BrainPy only support: {list(name2method.keys())}')

  if f is None:
    return lambda f: name2method[method](f, **kwargs)
  else:
    return name2method[method](f, **kwargs)


def set_default_fdeint(method):
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
  _DEFAULT_ODE_METHOD = method


def get_default_fdeint():
  """Get the default ODE numerical integrator method.

  Returns
  -------
  method : str
      The default numerical integrator method.
  """
  return _DEFAULT_DDE_METHOD


def register_fde_integrator(name, integrator):
  """Register a new ODE integrator.

  Parameters
  ----------
  name: ste
    The integrator name.
  integrator: type
    The integrator.
  """
  if name in name2method:
    raise ValueError(f'"{name}" has been registered in ODE integrators.')
  if not issubclass(integrator, FDEIntegrator):
    raise ValueError(f'"integrator" must be an instance of {FDEIntegrator.__name__}')
  name2method[name] = integrator


def get_supported_methods():
  """Get all supported numerical methods for DDEs."""
  return list(name2method.keys())
