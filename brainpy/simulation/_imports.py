# -*- coding: utf-8 -*-

try:
  import jax
  from brainpy.math import jax as mjax

except ModuleNotFoundError:
  jax = None
  mjax = None

__all__ = [
  'jax',
  'mjax',
]
