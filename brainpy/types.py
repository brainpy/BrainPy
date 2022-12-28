# -*- coding: utf-8 -*-

from typing import TypeVar, Tuple, Union, Callable

import jax.numpy as jnp
import numpy as np

from brainpy.math.ndarray import Array, Variable, TrainVar
from brainpy import connect as conn
from brainpy import initialize as init

__all__ = [
  'ArrayType', 'Parameter', 'PyTree',
  'Shape', 'Initializer',
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
Connector = Union[conn.Connector, Array, Variable, jnp.ndarray, np.ndarray]
Initializer = Union[init.Initializer, Callable, Array, Variable, jnp.ndarray, np.ndarray]

