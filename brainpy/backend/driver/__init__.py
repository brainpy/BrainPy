# -*- coding: utf-8 -*-

from brainpy.backend.driver.numpy import NumpyDSDriver
from brainpy.backend.driver.numpy import NumpyDiffIntDriver
from brainpy.simulation.drivers import BaseDSDriver
from brainpy.simulation.drivers import BaseDiffIntDriver

__all__ = [
  'switch_to',
  'set_buffer',
  'get_buffer',
  'get_ds_driver',
  'get_diffint_driver',

  'BUFFER',
]

DIFFINT_DRIVER = NumpyDiffIntDriver
DS_DRIVER = NumpyDSDriver
BUFFER = {}


def switch_to(backend):
  buffer = get_buffer(backend)

  global DS_DRIVER, DIFFINT_DRIVER
  if backend in ['numpy']:
    DS_DRIVER = buffer.get('ds', None) or NumpyDSDriver
    DIFFINT_DRIVER = buffer.get('diffint', None) or NumpyDiffIntDriver

  elif backend in ['numba', 'numba-parallel']:
    from brainpy.backend.driver import numba

    if backend == 'numba':
      numba.set_numba_profile(nogil=False, parallel=False)
    else:
      numba.set_numba_profile(nogil=True, parallel=True)

    DS_DRIVER = buffer.get('ds', None) or numba.NumbaDSDriver
    DIFFINT_DRIVER = buffer.get('diffint', None) or numba.NumbaDiffIntDriver

  elif backend in ['jax']:
    from brainpy.backend.driver import jax
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


def get_ds_driver():
  """Get the driver for dynamical systems.

  Returns
  -------
  node_driver
      The node driver.
  """
  return DS_DRIVER


def get_diffint_driver():
  """Get the current integration driver for differential equations.

  Returns
  -------
  diffint_driver
      The integration driver.
  """
  return DIFFINT_DRIVER

