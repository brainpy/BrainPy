# -*- coding: utf-8 -*-

from typing import TypeVar, Callable

import jax.numpy as jnp

import brainpy.math as bm
import brainpy.initialize as  init

__all__ = [
  'Tensor',
  'Initializer',
]

Tensor = TypeVar('Tensor', bm.JaxArray, jnp.ndarray)
Initializer = TypeVar('Initializer', bm.JaxArray, jnp.ndarray, init.Initializer, Callable)
