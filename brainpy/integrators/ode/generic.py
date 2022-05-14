# -*- coding: utf-8 -*-

from typing import Dict

from brainpy import math as bm
from .base import ODEIntegrator

__all__ = [
  'odeint',
  'set_default_odeint',
  'get_default_odeint',
  'register_ode_integrator',
  'get_supported_methods',
]

name2method = {
}

_DEFAULT_DDE_METHOD = 'euler'


def odeint(
    f=None,
    method=None,
    var_type=None,
    dt=None,
    name=None,
    show_code=False,
    state_delays: Dict[str, bm.AbstractDelay] = None,
    neutral_delays: Dict[str, bm.NeuTimeDelay] = None,
    **kwargs
):
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
    >>> @bp.odeint(method='rk4', dt=0.04)
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
    The type of the variable defined in the equation.
  dt: float
    The numerical integration precision.
  name: str
    The integrator node.
  state_delays: dict
    The state delay variable.
  show_code: bool
    Show the formated code.
  adaptive: bool
    The use adaptive mode.
  tol: float
    The tolerence to adapt new step size.

  Returns
  -------
  integral : ODEIntegrator
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
                                         show_code=show_code,
                                         state_delays=state_delays,
                                         neutral_delays=neutral_delays,
                                         **kwargs)
  else:
    return name2method[method](f,
                               var_type=var_type,
                               dt=dt,
                               name=name,
                               show_code=show_code,
                               state_delays=state_delays,
                               neutral_delays=neutral_delays,
                               **kwargs)


def set_default_odeint(method):
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


def get_default_odeint():
  """Get the default ODE numerical integrator method.

  Returns
  -------
  method : str
      The default numerical integrator method.
  """
  return _DEFAULT_DDE_METHOD


def register_ode_integrator(name, integrator):
  """Register a new ODE integrator.

  Parameters
  ----------
  name: ste
  integrator: type
  """
  if name in name2method:
    raise ValueError(f'"{name}" has been registered in ODE integrators.')
  if not issubclass(integrator, ODEIntegrator):
    raise ValueError(f'"integrator" must be an instance of {ODEIntegrator.__name__}')
  name2method[name] = integrator


def get_supported_methods():
  """Get all supported numerical methods for DDEs."""
  return list(name2method.keys())
