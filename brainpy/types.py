# -*- coding: utf-8 -*-

from typing import TypeVar, Tuple

import numpy as np
import jax.numpy as jnp


__all__ = [
  'Tensor', 'Parameter',

  'Shape',

  'Output', 'Monitor'
]

Parameter = TypeVar('Parameter', float, int, jnp.ndarray, 'JaxArray', 'Variable') # noqa
Tensor = TypeVar('Tensor', 'JaxArray', 'Variable', 'TrainVar', jnp.ndarray, np.ndarray) # noqa

Shape = TypeVar('Shape', int, Tuple[int, ...])

Output = TypeVar('Output')
Monitor = TypeVar('Monitor')

