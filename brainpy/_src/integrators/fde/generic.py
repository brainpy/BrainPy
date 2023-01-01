# -*- coding: utf-8 -*-

from .base import FDEIntegrator

__all__ = [
  'fdeint',
  'set_default_fdeint',
  'get_default_fdeint',
  'register_fde_integrator',
  'get_supported_methods',
]

name2method = {}

_DEFAULT_DDE_METHOD = 'l1'


def fdeint(
    alpha,
    num_memory,
    inits,
    f=None,
    method='l1',
    dt: str = None,
    name: str = None
):
  """Numerical integration for FDEs.

  Parameters
  ----------
  f : callable, function
    The derivative function.
  method : str
    The shortcut name of the numerical integrator.
  alpha: int, float, jnp.ndarray, bm.ndarray, sequence
    The fractional-order of the derivative function. Should be in the range of ``(0., 1.]``.
  num_memory: int
    The number of the memory length.
  inits: sequence
    A sequence of the initial values for variables.
  dt: float, int
    The numerical precision.
  name: str
    The integrator name.

  Returns
  -------
  integral : FDEIntegrator
      The numerical solver of `f`.
  """
  method = _DEFAULT_DDE_METHOD if method is None else method
  if method not in name2method:
    raise ValueError(f'Unknown FDE numerical method "{method}". Currently '
                     f'BrainPy supports: {list(name2method.keys())}')

  if f is None:
    return lambda f: name2method[method](f, dt=dt, name=name, inits=inits, num_memory=num_memory, alpha=alpha)
  else:
    return name2method[method](f, dt=dt, name=name, inits=inits, num_memory=num_memory, alpha=alpha)


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
    raise ValueError(f'"{name}" has been registered in FDE integrators.')
  if not issubclass(integrator, FDEIntegrator):
    raise ValueError(f'"integrator" must be an instance of {FDEIntegrator.__name__}')
  name2method[name] = integrator


def get_supported_methods():
  """Get all supported numerical methods for DDEs."""
  return list(name2method.keys())
