# -*- coding: utf-8 -*-

import logging

from brainpy import errors

try:
  import numba
  from brainpy.math.numpy import ast2numba
except ModuleNotFoundError:
  ast2numba = None
  numba = None

__all__ = [
  'jit',
  'vmap',
  'pmap',
]

logger = logging.getLogger('brainpy.math.numpy.compilation')


def jit(obj_or_func, nopython=True, fastmath=True, parallel=False, nogil=False, **kwargs):
  if ast2numba is None:
    raise errors.PackageMissingError('JIT compilation in numpy backend need Numba. '
                                     'Please install numba by: \n\n'
                                     '>>> pip install numba\n'
                                     '>>> # or \n'
                                     '>>> conda install numba')

  from brainpy.integrators import constants

  if callable(obj_or_func):  # function
    if hasattr(obj_or_func, '__name__') and obj_or_func.__name__.startswith(constants.DE_INT):
      return ast2numba.jit_integrator(obj_or_func,
                                      nopython=nopython,
                                      fastmath=fastmath,
                                      parallel=parallel,
                                      nogil=nogil)
    else:
      return numba.jit(obj_or_func,
                       nopython=nopython,
                       fastmath=fastmath,
                       parallel=parallel,
                       nogil=nogil)

  else:
    from brainpy.simulation.brainobjects.base import DynamicSystem
    from brainpy.simulation.brainobjects.neuron import NeuGroup
    from brainpy.simulation.brainobjects.synapse import TwoEndConn
    from brainpy.simulation.brainobjects.network import Network

    if isinstance(obj_or_func, DynamicSystem):
      if not isinstance(obj_or_func, (Network, NeuGroup, TwoEndConn)):
        raise errors.UnsupportedError(f'JIT compilation in numpy backend only supports '
                                      f'{NeuGroup.__name__}, {TwoEndConn.__name__}, and '
                                      f'{Network.__name__}, but we got {type(obj_or_func)}.')
      return ast2numba.jit_dynamic_system(obj_or_func,
                                          nopython=nopython,
                                          fastmath=fastmath,
                                          parallel=parallel,
                                          nogil=nogil)
    else:
      raise errors.UnsupportedError(f'JIT compilation in numpy backend does not support {type(obj_or_func)}.')


def vmap(obj_or_func, *args, **kwargs):
  _msg = 'Vectorize compilation is only supported in JAX backend, not available in numpy backend. \n' \
         'You can switch to JAX backend by `brainpy.math.use_backend("jax")`'
  logger.error(_msg)
  raise errors.BrainPyError(_msg)


def pmap(obj_or_func, *args, **kwargs):
  _msg = 'Parallel compilation is only supported in JAX backend, not available in numpy backend. \n' \
         'You can switch to JAX backend by `brainpy.math.use_backend("jax")`'
  logger.error(_msg)
  raise errors.BrainPyError(_msg)
