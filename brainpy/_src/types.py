# -*- coding: utf-8 -*-

from typing import TypeVar, Tuple, Union, Callable

import jax.numpy as jnp
import numpy as np

from brainpy._src.math.ndarray import Array
from brainpy._src.math.object_transform import Variable, TrainVar
from brainpy._src import connect as conn
from brainpy._src import initialize as init

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

