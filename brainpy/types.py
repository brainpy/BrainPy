# -*- coding: utf-8 -*-

from typing import TypeVar
import jax.numpy as jnp

import brainpy.math as bm

__all__ = [
  'Tensor',
]

Tensor = TypeVar('Tensor', bm.JaxArray, jnp.ndarray)


