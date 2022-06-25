# -*- coding: utf-8 -*-

from typing import TypeVar, Tuple

import numpy as np
import jax.numpy as jnp


__all__ = [
  'Tensor', 'Parameter',

  'Shape',

  'Output', 'Monitor'
]

# import brainpy.math as bm
# Tensor = TypeVar('Tensor', bm.JaxArray, jnp.ndarray, np.ndarray)
# Parameter = TypeVar('Parameter', float, int, jnp.ndarray, bm.JaxArray, bm.Variable)


Parameter = TypeVar('Parameter', float, int, jnp.ndarray, 'JaxArray', 'Variable')
Tensor = TypeVar('Tensor', 'JaxArray', 'Variable', 'TrainVar', jnp.ndarray, np.ndarray)

Shape = TypeVar('Shape', int, Tuple[int, ...])

Output = TypeVar('Output')
Monitor = TypeVar('Monitor')

