# -*- coding: utf-8 -*-

try:
  from jax import lax
  from jax import nn
  from jax import numpy as jnp
  from jax import scipy

  from brainpy.math.jax.ndarray import ndarray
  from brainpy.math.jax import random

except ModuleNotFoundError:
  lax = None
  nn = None
  jnp = None
  scipy = None
  ndarray = None
  random = None

__all__ = [
  'lax',
  'nn',
  'jnp',
  'scipy',
  'ndarray',
  'random',
]
