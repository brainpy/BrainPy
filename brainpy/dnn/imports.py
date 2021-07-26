# -*- coding: utf-8 -*-

try:
  import jax
  from brainpy.math import jax as jmath

except ModuleNotFoundError:
  jax = None
  jmath = None

__all__ = [
  'jax',
  'jmath',
]
