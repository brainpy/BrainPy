# -*- coding: utf-8 -*-

from . import drivers
from . import ops

__all__ = [
  'set',
  'set_class_keywords',
  'set_dt',
  'get_dt',
  'get_backend_name',

  'set_ops',
  'set_ops_from_module',
  'get_ds_driver',
  'get_diffint_driver',
]

from .ops import set_ops_from_module
from .ops import set_ops
from .drivers import get_ds_driver
from .drivers import get_diffint_driver

_dt = 0.1
BACKEND_NAME = 'numpy'
CLASS_KEYWORDS = ['self', 'cls']
SYSTEM_KEYWORDS = ['_dt', '_t', '_i']


def set(backend=None, dt=None):
  """Basic backend setting function.

  Using this function, users can set the backend they prefer. For backend
  which is unknown, users can provide `module_or_operations` to specify
  the operations needed. Also, users can customize the node runner, or the
  network runner, by providing the `node_runner` or `net_runner` keywords.
  The default numerical precision `dt` can also be set by this function.

  Parameters
  ----------
  backend : str
      The backend name.
  dt : float
      The numerical precision.
  """
  if dt is not None:
    set_dt(dt)
  if backend is not None:
    ops.switch_to(backend)
    drivers.switch_to(backend)
    global BACKEND_NAME
    BACKEND_NAME = backend


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


def set_dt(dt):
  """Set the numerical integrator precision.

  Parameters
  ----------
  dt : float
      Numerical integration precision.
  """
  assert isinstance(dt, float)
  global _dt
  _dt = dt


def get_dt():
  """Get the numerical integrator precision.

  Returns
  -------
  dt : float
      Numerical integration precision.
  """
  return _dt


def get_backend_name():
  """Get the current backend name.

  Returns
  -------
  backend : str
      The name of the current backend name.
  """
  return BACKEND_NAME
