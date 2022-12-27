# -*- coding: utf-8 -*-

from typing import TypeVar, Tuple

import jax.numpy as jnp
import numpy as np

from brainpy.math.ndarray import Array, Variable, TrainVar

__all__ = [
  'ArrayType', 'Parameter', 'PyTree',
  'Shape',
  'Output', 'Monitor'
]


# data
Parameter = TypeVar('Parameter', float, int, jnp.ndarray, 'Array', 'Variable') # noqa
ArrayType = TypeVar('ArrayType', Array, Variable, TrainVar, jnp.ndarray, np.ndarray) # noqa
Array = ArrayType # noqa
PyTree = TypeVar('PyTree') # noqa

# shape
Shape = TypeVar('Shape', int, Tuple[int, ...]) # noqa

# component
Output = TypeVar('Output') # noqa
Monitor = TypeVar('Monitor') # noqa

