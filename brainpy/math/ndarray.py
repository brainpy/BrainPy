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
from typing import Any

import brainunit as u
import jax
import numpy as np
from jax import numpy as jnp
from jax.dtypes import canonicalize_dtype
from jax.tree_util import register_pytree_node_class

from brainpy._errors import MathError
from .defaults import defaults

bm = None

__all__ = [
    'Array', 'Array', 'ndarray', 'JaxArray',  # alias of Array
    'ShardedArray',
]

# Ways to change values in a zero-dimensional array
# -----
# Reference: https://stackoverflow.com/questions/56954714/how-do-i-assign-to-a-zero-dimensional-numpy-array
#
#   >>> x = np.array(10)
# 1. index the original array with ellipsis or an empty tuple
#    >>> x[...] = 2
#    >>> x[()] = 2

_all_slice = slice(None, None, None)


def _check_input_array(array):
    if isinstance(array, Array):
        return array.value
    elif isinstance(array, np.ndarray):
        return jnp.asarray(array)
    else:
        return array


def _return(a):
    if defaults.numpy_func_return == 'bp_array' and isinstance(a, jax.Array) and a.ndim > 0:
        return Array(a)
    return a


def _as_jax_array_(obj):
    return obj.value if isinstance(obj, Array) else obj


def _check_out(out):
    if not isinstance(out, Array):
        raise TypeError(f'out must be an instance of brainpy Array. But got {type(out)}')


def _get_dtype(v):
    if hasattr(v, 'dtype'):
        dtype = v.dtype
    else:
        dtype = canonicalize_dtype(type(v))
    return dtype


@register_pytree_node_class
class Array(u.CustomArray):
    """Multiple-dimensional array in BrainPy.

    Compared to ``jax.Array``, :py:class:`~.Array` has the following advantages:

    - In-place updating is supported.

    >>> import brainpy.math as bm
    >>> a = bm.asarray([1, 2, 3.])
    >>> a[0] = 10.

    - Keep sharding constraints during computation.

    - More dense array operations with PyTorch syntax.

    """

    __slots__ = ('_value',)

    def __init__(self, value, dtype: Any = None):
        # array value
        if isinstance(value, Array):
            value = value.value
        elif isinstance(value, (tuple, list, np.ndarray)):
            value = jnp.asarray(value)
        if dtype is not None:
            value = jnp.asarray(value, dtype=dtype)
        self._value = value

    def __repr__(self) -> str:
        print_code = repr(self.value)
        if ', dtype' in print_code:
            print_code = print_code.split(', dtype')[0] + ')'
        prefix = f'{self.__class__.__name__}'
        prefix2 = f'{self.__class__.__name__}(value='
        if '\n' in print_code:
            lines = print_code.split("\n")
            blank1 = " " * len(prefix2)
            lines[0] = prefix2 + lines[0]
            for i in range(1, len(lines)):
                lines[i] = blank1 + lines[i]
            lines[-1] += ","
            blank2 = " " * (len(prefix) + 1)
            lines.append(f'{blank2}dtype={self.dtype})')
            print_code = "\n".join(lines)
        else:
            print_code = prefix2 + print_code + f', dtype={self.dtype})'
        return print_code

    def tree_flatten(self):
        return (self.value,), None

    @classmethod
    def tree_unflatten(cls, aux_data, flat_contents):
        return cls(*flat_contents)

        # ins = object.__new__(cls)
        # ins._value = flat_contents[0]
        # return ins

    @property
    def data(self):
        return self.value

    @data.setter
    def data(self, value):
        self.value = value

    @property
    def value(self):
        # return the value
        return self._value

    @value.setter
    def value(self, value):
        self_value = self._value

        if isinstance(value, Array):
            value = value.value
        elif isinstance(value, np.ndarray):
            value = jnp.asarray(value)
        elif isinstance(value, jax.Array):
            pass
        else:
            value = jnp.asarray(value)
        # # check
        # if value.shape != self_value.shape:
        #     raise MathError(f"The shape of the original data is {self_value.shape}, "
        #                     f"while we got {value.shape}.")
        # if value.dtype != self_value.dtype:
        #     raise MathError(f"The dtype of the original data is {self_value.dtype}, "
        #                     f"while we got {value.dtype}.")
        self._value = value

    def update(self, value):
        """Update the value of this Array.
        """
        self.value = value

    def __array__(self, dtype=None):
        """Support ``numpy.array()`` and ``numpy.asarray()`` functions."""
        return np.asarray(self.value, dtype=dtype)

    def __jax_array__(self):
        return self.value

    def as_variable(self):
        """As an instance of Variable."""
        global bm
        if bm is None: from brainpy import math as bm
        return bm.Variable(self)

    # ----------------------- #
    #       JAX methods       #
    # ----------------------- #

    @property
    def at(self):
        return self.value.at

    def block_host_until_ready(self, *args):
        return self.value.block_host_until_ready(*args)

    def block_until_ready(self, *args):
        return self.value.block_until_ready(*args)

    def device(self):
        return self.value.device()

    @property
    def device_buffer(self):
        return self.value.device_buffer

    def fill_(self, fill_value):
        """Fill the array with a scalar value.

        Args:
          fill_value: the scalar value to fill the array.
        """
        if isinstance(fill_value, Array):
            fill_value = fill_value.value
        elif isinstance(fill_value, np.ndarray):
            fill_value = jnp.asarray(fill_value)
        elif isinstance(fill_value, jax.Array):
            pass
        else:
            fill_value = jnp.asarray(fill_value)
        # check
        if fill_value.shape != ():
            raise MathError(f"The shape of the fill value must be (), "
                            f"while we got {fill_value.shape}.")
        self.value = jnp.full(self.shape, fill_value, dtype=self.dtype)
        return self


setattr(Array, "__array_priority__", 100)

JaxArray = Array
ndarray = Array


@register_pytree_node_class
class ShardedArray(Array):
    """The sharded array, which stores data across multiple devices.

    A drawback of sharding is that the data may not be evenly distributed on shards.

    Args:
      value: the array value.
      dtype: the array type.
      keep_sharding: keep the array sharding information using ``jax.lax.with_sharding_constraint``. Default True.
    """

    __slots__ = ('_value', '_keep_sharding')

    def __init__(self, value, dtype: Any = None, *, keep_sharding: bool = True):
        super().__init__(value, dtype)
        self._keep_sharding = keep_sharding

    @property
    def value(self):
        """The value stored in this array.

        Returns:
          The stored data.
        """
        v = self._value
        # keep sharding constraints
        if self._keep_sharding and hasattr(v, 'sharding') and (v.sharding is not None):
            return jax.lax.with_sharding_constraint(v, v.sharding)
        # return the value
        return v

    @value.setter
    def value(self, value):
        self_value = self._check_tracer()

        if isinstance(value, Array):
            value = value.value
        elif isinstance(value, np.ndarray):
            value = jnp.asarray(value)
        elif isinstance(value, jax.Array):
            pass
        else:
            value = jnp.asarray(value)
        # check
        if value.shape != self_value.shape:
            raise MathError(f"The shape of the original data is {self_value.shape}, "
                            f"while we got {value.shape}.")
        if value.dtype != self_value.dtype:
            raise MathError(f"The dtype of the original data is {self_value.dtype}, "
                            f"while we got {value.dtype}.")
        self._value = value
