# -*- coding: utf-8 -*-

try:
  import jax
  from brainpy.math import jax as mjax

except (ModuleNotFoundError, ImportError):
  jax = None
  mjax = None


__all__ = [
  'jax',
  'mjax',
]
