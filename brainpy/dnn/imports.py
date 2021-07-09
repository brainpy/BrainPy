# -*- coding: utf-8 -*-

try:
  import jax
  from brainpy.math import jax as jax_math

except ModuleNotFoundError:
  jax = None
  jax_math = None

__all__ = [
  'jax',
  'jax_math',
]
