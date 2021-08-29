# -*- coding: utf-8 -*-

import logging

from brainpy import errors

__all__ = [
  'jit',
  'vmap',
  'pmap',
]

logger = logging.getLogger('brainpy.math.numpy.compilation')


def jit(obj_or_func, *args, **kwargs):
  logger.warning('JIT compilation in numpy backend can not be available right now.')
  return obj_or_func


def vmap(obj_or_func, *args, **kwargs):
  _msg = 'Vectorize compilation is only supported in JAX backend, not available in numpy backend. \n' \
         'You can switch to JAX backend by `brainpy.math.use_backend("jax")`'
  logger.error(_msg)
  raise errors.ModelUseError(_msg)


def pmap(obj_or_func, *args, **kwargs):
  _msg = 'Parallel compilation is only supported in JAX backend, not available in numpy backend. \n' \
         'You can switch to JAX backend by `brainpy.math.use_backend("jax")`'
  logger.error(_msg)
  raise errors.ModelUseError(_msg)
