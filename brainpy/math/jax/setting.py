# -*- coding: utf-8 -*-

from jax.config import config

__all__ = [
  'enable_x64',
]


def enable_x64(mode=True):
  assert mode in [True, False]
  config['JAX_ENABLE_X64'] = mode


