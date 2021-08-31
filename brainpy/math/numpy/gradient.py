# -*- coding: utf-8 -*-

import logging

from brainpy import errors

__all__ = [
  'grad', 'value_and_grad',
]

logger = logging.getLogger('brainpy.math.numpy.gradient')


def grad(func, *args, **kwargs):
  _msg = '"grad" is only supported in JAX backend, not available in numpy backend. \n' \
         'You can switch to JAX backend by `brainpy.math.use_backend("jax")`'
  logger.error(_msg)
  raise errors.BrainPyError(_msg)


def value_and_grad(func, *args, **kwargs):
  _msg = '"value_and_grad" is only supported in JAX backend, not available in numpy backend. \n' \
         'You can switch to JAX backend by `brainpy.math.use_backend("jax")`'
  logger.error(_msg)
  raise errors.BrainPyError(_msg)
