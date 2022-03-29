# -*- coding: utf-8 -*-

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

name2method = {
}

_DEFAULT_DDE_METHOD = 'euler'


def ddeint(f=None,
           method='euler',
           var_type: str = None,
           dt: Union[float, int] = None,
           name: str = None,
           show_code: bool = False,
           state_delays: Dict[str, bm.TimeDelay] = None,
           neutral_delays: Dict[str, bm.NeutralDelay] = None,
           **kwargs):
  """Numerical integration for ODEs.

  Examples
  --------

  .. plot::
    :include-source: True

    >>> import brainpy as bp
    >>> import matplotlib.pyplot as plt
    >>>
    >>> a=0.7;  b=0.8;  tau=12.5;  Vth=1.9
    >>> V = 0;  w = 0  # initial values
    >>>
    >>> @bp.ddeint(method='rk4', dt=0.04)
    >>> def integral(V, w, t, Iext):
    >>>   dw = (V + a - b * w) / tau
    >>>   dV = V - V * V * V / 3 - w + Iext
    >>>   return dV, dw
    >>>
    >>> hist_V = []
    >>> for t in bp.math.arange(0, 100, integral.dt):
    >>>     V, w = integral(V, w, t, 0.5)
    >>>     hist_V.append(V)
    >>> plt.plot(bp.math.arange(0, 100, integral.dt), hist_V)
    >>> plt.show()

  Parameters
  ----------
  f : callable, function
    The derivative function.
  method : str
    The shortcut name of the numerical integrator.
  var_type: str
    Variable type in the defined function.
  dt: float, int
    The time precission for integration.
  name: str
    The name.
  show_code: bool
    Whether show the formartted codes.
  state_delays: dict
    The state delay variables.
  neutral_delays: dict
    The neutral delay variable.

  Returns
  -------
  integral : DDEIntegrator
      The numerical solver of `f`.
  """
  method = _DEFAULT_DDE_METHOD if method is None else method
  if method not in name2method:
    raise ValueError(f'Unknown ODE numerical method "{method}". Currently '
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
