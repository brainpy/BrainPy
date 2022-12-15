# -*- coding: utf-8 -*-

from typing import TypeVar, Tuple

import numpy as np
import jax.numpy as jnp


__all__ = [
  'ArrayType', 'Parameter',

  'Shape',

  'Output', 'Monitor'
]

Parameter = TypeVar('Parameter', float, int, jnp.ndarray, 'Array', 'Variable') # noqa
ArrayType = TypeVar('Array', 'Variable', 'TrainVar', jnp.ndarray, np.ndarray) # noqa
Array = ArrayType

Shape = TypeVar('Shape', int, Tuple[int, ...])

Output = TypeVar('Output')
Monitor = TypeVar('Monitor')

