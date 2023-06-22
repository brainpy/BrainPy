# -*- coding: utf-8 -*-

import numbers
from typing import TypeVar, Tuple, Union, Callable, Sequence

import jax
import numpy as np

from brainpy._src import connect as conn
from brainpy._src import initialize as init
from brainpy._src.math.ndarray import Array
from brainpy._src.math.object_transform import Variable, TrainVar

__all__ = [
  'ArrayType', 'Parameter', 'PyTree',
  'Shape', 'Initializer',
  'Output', 'Monitor', 'Sharding',
]


# data
Parameter = TypeVar('Parameter', numbers.Number, jax.Array, 'Array', 'Variable') # noqa
ArrayType = TypeVar('ArrayType', Array, Variable, TrainVar, jax.Array, np.ndarray) # noqa
Array = ArrayType # noqa
PyTree = TypeVar('PyTree') # noqa

# shape
Shape = TypeVar('Shape', int, Tuple[int, ...]) # noqa

# component
Output = TypeVar('Output') # noqa
Monitor = TypeVar('Monitor') # noqa
Connector = Union[conn.Connector, Array, Variable, jax.Array, np.ndarray]
Initializer = Union[init.Initializer, Callable, Array, Variable, jax.Array, np.ndarray]

Sharding = Union[Sequence[str], jax.sharding.Sharding, jax.Device]

