# -*- coding: utf-8 -*-

from typing import TypeVar, Tuple

import numpy as np
import jax.numpy as jnp


__all__ = [
  'Array', 'Parameter',

  'Shape',

  'Output', 'Monitor'
]

Parameter = TypeVar('Parameter', float, int, jnp.ndarray, 'JaxArray', 'Variable') # noqa
Array = TypeVar('Array', 'JaxArray', 'Variable', 'TrainVar', jnp.ndarray, np.ndarray) # noqa

Shape = TypeVar('Shape', int, Tuple[int, ...])

Output = TypeVar('Output')
Monitor = TypeVar('Monitor')

