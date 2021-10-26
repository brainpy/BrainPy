# -*- coding: utf-8 -*-

"""
Numerical methods for stochastic differential equations.
"""

from .euler_and_milstein import *
from .srk_scalar import *

name2method = {
  'euler': Euler,
  'heun': Heun,
  'milstein': Milstein,
  'exponential_euler': ExponentialEuler, 'exp_euler': ExponentialEuler,
  'srk1w1': SRK1W1, 'srk2w1': SRK2W1
}


def sdeint(f=None, g=None, method='euler', **kwargs):
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
  if method not in name2method:
    raise ValueError(f'Unknown SDE numerical method "{method}". Currently '
                     f'BrainPy only support: {list(name2method.keys())}')

  if f is not None and g is not None:
    return name2method[method](f=f, g=g, **kwargs)

  elif f is not None:
    return lambda g: name2method[method](f=f, g=g, **kwargs)

  elif g is not None:
    return lambda f: name2method[method](f=f, g=g, **kwargs)

  else:
    raise ValueError('Must provide "f" or "g".')
