# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numbers
from typing import TypeVar, Tuple, Union, Callable, Sequence

import jax
import numpy as np

from brainpy import connect as conn
from brainpy import initialize as init
from brainpy.math.ndarray import Array
from brainpy.math.object_transform import Variable, TrainVar

__all__ = [
    'ArrayType', 'Parameter', 'PyTree',
    'Shape', 'Initializer',
    'Output', 'Monitor', 'Sharding',
]

# data
Parameter = TypeVar('Parameter', numbers.Number, jax.Array, 'Array', 'Variable')  # noqa
ArrayType = TypeVar('ArrayType', Array, Variable, TrainVar, jax.Array, np.ndarray)  # noqa
Array = ArrayType  # noqa
PyTree = TypeVar('PyTree')  # noqa

# shape
Shape = TypeVar('Shape', int, Tuple[int, ...])  # noqa

# component
Output = TypeVar('Output')  # noqa
Monitor = TypeVar('Monitor')  # noqa
Connector = Union[conn.Connector, Array, Variable, jax.Array, np.ndarray]
Initializer = Union[init.Initializer, Callable, Array, Variable, jax.Array, np.ndarray]

Sharding = Union[Sequence[str], jax.sharding.Sharding, jax.Device]
