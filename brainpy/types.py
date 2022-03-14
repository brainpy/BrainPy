# -*- coding: utf-8 -*-

from typing import TypeVar, Tuple

import jax.numpy as jnp

import brainpy.math as bm

__all__ = [
  'Tensor',
  'Parameter',
  'Shape',
]

Tensor = TypeVar('Tensor', bm.JaxArray, jnp.ndarray)
Parameter = TypeVar('Parameter', bm.JaxArray, jnp.ndarray, float, int, bm.Variable)
Shape = TypeVar('Shape', int, Tuple[int, ...])
