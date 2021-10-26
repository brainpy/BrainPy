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

_name2method = {
  # explicit RK
  'euler': Euler,
  'midpoint': MidPoint,
  'heun2': Heun2,
  'ralston2': Ralston2,
  'rk2': RK2,
  'rk3': RK3,
  'heun3': Heun3,
  'ralston3': Ralston3,
  'ssprk3': SSPRK3,
  'rk4': RK4,
  'ralston4': Ralston4,
  'rk4_38rule': RK4Rule38,

  # adaptive RK
  'rkf12': RKF12,
  'rkf45': RKF45,
  'rkdp': DormandPrince, 'dp': DormandPrince,
  'ck': DormandPrince,
  'bs': BogackiShampine,
  'heun_euler': HeunEuler,

  # exponential euler
  'exponential_euler': ExponentialEuler, 'exp_euler': ExponentialEuler,
}


def odeint(f=None, method='euler', **kwargs):
  """Numerical integration for ODEs.

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
  if method not in _name2method:
    raise ValueError(f'Unknown ODE numerical method "{method}". Currently '
                     f'BrainPy only support: {list(_name2method.keys())}')

  if f is None:
    return lambda f: _name2method[method](f, **kwargs)
  else:
    return _name2method[method](f, **kwargs)
