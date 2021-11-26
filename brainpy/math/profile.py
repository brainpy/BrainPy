# -*- coding: utf-8 -*-

from typing import List

CLASS_KEYWORDS: List[str] = ['self', 'cls']
SYSTEM_KEYWORDS: List[str] = ['_dt', '_t', '_i']

__all__ = [
  'set_dt',
  'get_dt',
]

# numerical precision
# --------------------------

__dt = 0.1


def set_dt(dt):
  """Set the numerical integrator precision.

  Parameters
  ----------
  dt : float
      Numerical integration precision.
  """
  assert isinstance(dt, float), f'"dt" must a float, but we got {dt}'
  global __dt
  __dt = dt


def get_dt():
  """Get the numerical integrator precision.

  Returns
  -------
  dt : float
      Numerical integration precision.
  """
  return __dt


def set_class_keywords(*args):
  """Set the keywords for class specification.

  For example:

  >>> class A(object):
  >>>    def __init__(cls):
  >>>        pass
  >>>    def f(self, ):
  >>>        pass

  In this case, I use "cls" to denote the "self". So, I can set this by

  >>> set_class_keywords('cls', 'self')

  """
  global CLASS_KEYWORDS
  CLASS_KEYWORDS = list(args)


def get_class_keywords():
  return CLASS_KEYWORDS
