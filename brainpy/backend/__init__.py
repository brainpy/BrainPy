# -*- coding: utf-8 -*-

from brainpy import math
from brainpy.backend import jax, numba, numpy


__all__ = [
  'set_class_keywords',
  'get_ds_driver',
  'get_diffint_driver',
]


_backend_to_drivers = {
  'numpy': {
    'diffint': numpy.NumpyDiffIntDriver,
    'ds': numpy.NumpyDSDriver
  },
  'numba': {
    'diffint': numba.NumbaDiffIntDriver,
    'ds': numba.NumbaDSDriver
  },
  'jax': {
    'diffint': jax.JaxDiffIntDriver,
    'ds': jax.JaxDSDriver
  },
}

CLASS_KEYWORDS = ['self', 'cls']
SYSTEM_KEYWORDS = ['_dt', '_t', '_i']


def switch_to(backend):
  buffer = get_buffer(backend)

  global DS_DRIVER, DIFFINT_DRIVER
  if backend in ['numpy']:
    DS_DRIVER = buffer.get('ds', None) or numpy.NumpyDSDriver
    DIFFINT_DRIVER = buffer.get('diffint', None) or numpy.NumpyDiffIntDriver

  elif backend in ['numba', 'numba-parallel']:

    if backend == 'numba':
      numba.set_numba_profile(nogil=False, parallel=False)
    else:
      numba.set_numba_profile(nogil=True, parallel=True)

    DS_DRIVER = buffer.get('ds', None) or numba.NumbaDSDriver
    DIFFINT_DRIVER = buffer.get('diffint', None) or numba.NumbaDiffIntDriver

  elif backend in ['jax']:
    DS_DRIVER = buffer.get('ds', None) or jax.JaxDSDriver
    DIFFINT_DRIVER = buffer.get('diffint', None) or jax.JaxDiffIntDriver

  else:
    if 'ds' not in buffer:
      raise ValueError(f'"{backend}" is an unknown backend, should '
                       f'set DS buffer by "brainpy.drivers.set_buffer'
                       f'(backend, ds_driver=SomeDSDriver)"')
    if 'diffint' not in buffer:
      raise ValueError(f'"{backend}" is an unknown backend, should '
                       f'set integrator wrapper by "brainpy.drivers.'
                       f'set_buffer(backend, diffint_driver=SomeDriver)"')
    DS_DRIVER = buffer.get('ds')
    DIFFINT_DRIVER = buffer.get('diffint')


def set_buffer(backend, ds_driver=None, diffint_driver=None):
  from brainpy.simulation.drivers import BaseDSDriver, BaseDiffIntDriver

  global BUFFER
  if backend not in BUFFER:
    BUFFER[backend] = dict()

  if ds_driver is not None:
    assert BaseDSDriver in ds_driver.__bases__
    BUFFER[backend]['ds'] = ds_driver
  if diffint_driver is not None:
    assert BaseDiffIntDriver in diffint_driver.__bases__
    BUFFER[backend]['diffint'] = diffint_driver


def get_buffer(backend):
  return BUFFER.get(backend, dict())


def get_ds_driver(backend=None):
  """Get the driver for dynamical systems.

  Returns
  -------
  node_driver
      The node driver.
  """
  if backend is not None:
    return _backend_to_drivers[backend]['ds']
  else:
    return _backend_to_drivers[math.get_backend_name()]['ds']


def get_diffint_driver(backend=None):
  """Get the current integration driver for differential equations.

  Returns
  -------
  diffint_driver
      The integration driver.
  """
  if backend is not None:
    return _backend_to_drivers[backend]['diffint']
  else:
    return _backend_to_drivers[math.get_backend_name()]['diffint']


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
