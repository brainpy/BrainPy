# -*- coding: utf-8 -*-

import logging

from brainpy import errors

try:
  import numba
  from brainpy.math.numpy import ast2numba
except ModuleNotFoundError:
  ast2numba = None
  numba = None

DE_INT = None
DynamicSystem = None

__all__ = [
  'jit',
  'vmap',
  'pmap',
]

logger = logging.getLogger('brainpy.math.numpy.compilation')


def jit(obj_or_func, nopython=True, fastmath=True, parallel=False, nogil=False, show_code=False, **kwargs):
  # checking
  if ast2numba is None or numba is None:
    raise errors.PackageMissingError('JIT compilation in numpy backend need Numba. '
                                     'Please install numba via: \n\n'
                                     '>>> pip install numba\n'
                                     '>>> # or \n'
                                     '>>> conda install numba')
  global DE_INT, DynamicSystem
  if DE_INT is None:
    from brainpy.integrators.constants import DE_INT
  if DynamicSystem is None:
    from brainpy.simulation.brainobjects.base import DynamicSystem

  # JIT compilation
  if callable(obj_or_func):
    # integrator
    if hasattr(obj_or_func, '__name__') and obj_or_func.__name__.startswith(DE_INT):
      return ast2numba.jit_integrator(obj_or_func,
                                      nopython=nopython,
                                      fastmath=fastmath,
                                      parallel=parallel,
                                      nogil=nogil,
                                      show_code=show_code)
    else:
      # native function
      return numba.jit(obj_or_func,
                       nopython=nopython,
                       fastmath=fastmath,
                       parallel=parallel,
                       nogil=nogil)

  else:
    # dynamic system
    if not isinstance(obj_or_func, DynamicSystem):
      raise errors.UnsupportedError(f'JIT compilation in numpy backend only supports '
                                    f'{DynamicSystem.__name__}, but we got {type(obj_or_func)}.')
    return ast2numba.jit_dynamic_system(obj_or_func,
                                        nopython=nopython,
                                        fastmath=fastmath,
                                        parallel=parallel,
                                        nogil=nogil,
                                        show_code=show_code)


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
