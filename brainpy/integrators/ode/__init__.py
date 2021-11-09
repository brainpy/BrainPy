# -*- coding: utf-8 -*-

"""
Numerical methods for ordinary differential equations.
"""

from . import adaptive_rk
from . import explicit_rk
from . import exponential
from .base import *
from .adaptive_rk import *
from .explicit_rk import *
from .exponential import *

name2method = {
  # explicit RK
  'euler': Euler, 'Euler': Euler,
  'midpoint': MidPoint, 'MidPoint': MidPoint,
  'heun2': Heun2, 'Heun2': Heun2,
  'ralston2': Ralston2, 'Ralston2': Ralston2,
  'rk2': RK2, 'RK2': RK2,
  'rk3': RK3, 'RK3': RK3,
  'heun3': Heun3, 'Heun3': Heun3,
  'ralston3': Ralston3, 'Ralston3': Ralston3,
  'ssprk3': SSPRK3, 'SSPRK3': SSPRK3,
  'rk4': RK4, 'RK4': RK4,
  'ralston4': Ralston4, 'Ralston4': Ralston4,
  'rk4_38rule': RK4Rule38, 'RK4Rule38': RK4Rule38,

  # adaptive RK
  'rkf12': RKF12, 'RKF12': RKF12,
  'rkf45': RKF45, 'RKF45': RKF45,
  'rkdp': DormandPrince, 'dp': DormandPrince, 'DormandPrince': DormandPrince,
  'ck': CashKarp, 'CashKarp': CashKarp,
  'bs': BogackiShampine, 'BogackiShampine': BogackiShampine,
  'heun_euler': HeunEuler, 'HeunEuler': HeunEuler,

  # exponential euler
  'exponential_euler': ExponentialEuler, 'exp_euler': ExponentialEuler, 'ExponentialEuler': ExponentialEuler,
}

_DEFAULT_ODE_METHOD = 'euler'


def odeint(f=None, method='euler', **kwargs):
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

  Returns
  -------
  integral : callable
      The numerical solver of `f`.
  """
  method = _DEFAULT_ODE_METHOD if method is None else method
  if method not in name2method:
    raise ValueError(f'Unknown ODE numerical method "{method}". Currently '
                     f'BrainPy only support: {list(name2method.keys())}')

  if f is None:
    return lambda f: name2method[method](f, **kwargs)
  else:
    return name2method[method](f, **kwargs)


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
