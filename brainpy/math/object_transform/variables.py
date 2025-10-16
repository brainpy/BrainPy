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
from typing import Optional, Any, Sequence

import brainstate
import jax
import numpy as np
from brainstate._state import record_state_value_read, record_state_value_write
from jax import numpy as jnp
from jax.dtypes import canonicalize_dtype
from jax.tree_util import register_pytree_node_class

from brainpy._errors import MathError
from brainpy.math.ndarray import Array
from brainpy.math.sharding import BATCH_AXIS

__all__ = [
    'Variable',
    'TrainVar',
    'Parameter',
    'VariableView',

    'VarList', 'var_list',
    'VarDict', 'var_dict',
]


@register_pytree_node_class
class Variable(brainstate.State, Array):
    """The pointer to specify the dynamical variable.

    Initializing an instance of ``Variable`` by two ways:

    >>> import brainpy.math as bm
    >>> # 1. init a Variable by the concreate data
    >>> v1 = bm.Variable(bm.zeros(10))
    >>> # 2. init a Variable by the data shape
    >>> v2 = bm.Variable(10)

    Note that when initializing a `Variable` by the data shape,
    all values in this `Variable` will be initialized as zeros.

    Args:
      value_or_size: Shape, Array, int. The value or the size of the value.
      dtype: Any. The type of the data.
      batch_axis: optional, int. The batch axis.
      axis_names: sequence of str. The name for each axis.
    """

    def __init__(
        self,
        value_or_size: Any,
        dtype: type = None,
        batch_axis: int = None,
        *,
        axis_names: Optional[Sequence[str]] = None,
    ):
        if isinstance(value_or_size, int):
            value = jnp.zeros(value_or_size, dtype=dtype)
        elif isinstance(value_or_size, (tuple, list)) and all([isinstance(s, int) for s in value_or_size]):
            value = jnp.zeros(value_or_size, dtype=dtype)
        else:
            value = value_or_size

        if isinstance(value, Array):
            value = value.value
        Array.__init__(self, value, dtype=dtype)
        brainstate.State.__init__(self, value)

        # check batch axis
        if isinstance(value, Variable):
            if value.batch_axis is not None and batch_axis is not None:
                if batch_axis != value.batch_axis:
                    raise ValueError(f'"batch_axis" is not consistent. Got batch_axis in the given value '
                                     f'is {value.batch_axis}, but the specified batch_axis is {batch_axis}')
            batch_axis = value.batch_axis

        # assign batch axis
        self._batch_axis = batch_axis
        if batch_axis is not None:
            if batch_axis >= np.ndim(self._value):
                raise MathError(f'This variables has {np.ndim(self._value)} dimension, '
                                f'but the batch axis is set to be {batch_axis}.')

        # ready to trace the variable
        if axis_names is not None:
            if len(axis_names) + 1 == self.ndim:
                axis_names = list(axis_names)
                axis_names.insert(self.batch_axis, BATCH_AXIS)
            assert len(axis_names) == self.ndim
            axis_names = tuple(axis_names)
        self.axis_names = axis_names

    @property
    def size_without_batch(self):
        if self.batch_axis is None:
            return self.size
        else:
            sizes = self.size
            return sizes[:self.batch_size] + sizes[self.batch_axis + 1:]

    @property
    def batch_axis(self) -> Optional[int]:
        return self._batch_axis

    @batch_axis.setter
    def batch_axis(self, val):
        raise ValueError(f'Cannot set "batch_axis" after creating a {self.__class__.__name__} instance.')

    @property
    def batch_size(self) -> Optional[int]:
        if self.batch_axis is None:
            return None
        else:
            return self.shape[self.batch_axis]

    @batch_size.setter
    def batch_size(self, val):
        raise ValueError(f'Cannot set "batch_size" manually.')

    def _ensure_value_exists(self):
        pass

    @property
    def value(self):
        self._ensure_value_exists()
        record_state_value_read(self)
        return self._read_value()

    @value.setter
    def value(self, v):
        _value = self.value
        ext_shape = jnp.shape(v)
        int_shape = jnp.shape(_value)
        if self._batch_axis is not None:
            ext_shape = ext_shape[:self._batch_axis] + ext_shape[self._batch_axis + 1:]
            int_shape = int_shape[:self._batch_axis] + int_shape[self._batch_axis + 1:]
        if ext_shape != int_shape:
            error = f"The shape of the original data is {int_shape}, while we got {ext_shape}"
            error += f' with batch_axis={self._batch_axis}.'
            raise MathError(error)
        ext_dtype = _get_dtype(v)
        int_dtype = self.dtype
        if ext_dtype != int_dtype:
            raise MathError(f"The dtype of the original data is {int_dtype}, "
                            f"while we got {ext_dtype}.")
        if isinstance(v, Array):
            v = v.value
        elif isinstance(v, np.ndarray):
            v = jnp.asarray(v)
        else:
            v = v

        if isinstance(v, brainstate.State):  # value checking
            v = v.value
        self._check_value_tree(v)  # check the tree structure
        record_state_value_write(self)  # record the value by the stack (>= level)
        self._been_writen = True  # set the flag
        self._write_value(v)  # write the value


def _get_dtype(v):
    if hasattr(v, 'dtype'):
        dtype = v.dtype
    else:
        dtype = canonicalize_dtype(type(v))
    return dtype


def _as_jax_array_(obj):
    return obj.value if isinstance(obj, Array) else obj


@register_pytree_node_class
class TrainVar(Variable):
    """The pointer to specify the trainable variable.
    """

    def __init__(
        self,
        value_or_size: Any,
        dtype: type = None,
        batch_axis: int = None,
        *,
        axis_names: Optional[Sequence[str]] = None,
    ):
        super().__init__(
            value_or_size,
            dtype=dtype,
            batch_axis=batch_axis,
            axis_names=axis_names,
        )


@register_pytree_node_class
class Parameter(Variable):
    """The pointer to specify the parameter.
    """

    def __init__(
        self,
        value_or_size: Any,
        dtype: type = None,
        batch_axis: int = None,
        *,
        axis_names: Optional[Sequence[str]] = None,
    ):
        super().__init__(
            value_or_size,
            dtype=dtype,
            batch_axis=batch_axis,
            axis_names=axis_names,
        )


class VariableView(Variable):
    """A view of a Variable instance.

    This class is used to create a subset view of ``brainpy.math.Variable``.

    >>> import brainpy.math as bm
    >>> bm.random.seed(123)
    >>> origin = bm.Variable(bm.random.random(5))
    >>> view = bm.VariableView(origin, slice(None, 2, None))  # origin[:2]
    VariableView([0.02920651, 0.19066381], dtype=float32)

    ``VariableView`` can be used to update the subset of the original
    Variable instance, and make operations on this subset of the Variable.

    >>> view[:] = 1.
    >>> view
    VariableView([1., 1.], dtype=float32)
    >>> origin
    Variable([1.       , 1.       , 0.5482849, 0.6564884, 0.8446237], dtype=float32)
    >>> view + 10
    Array([11., 11.], dtype=float32)
    >>> view *= 10
    VariableView([10., 10.], dtype=float32)

    The above example demonstrates that the updating of an ``VariableView`` instance
    is actually made in the original ``Variable`` instance.

    Moreover, it's worthy to note that ``VariableView`` is not a PyTree.
    """
    _need_record = False

    def __init__(
        self,
        value: Variable,
        index: Any,
    ):
        self.index = jax.tree_util.tree_map(_as_jax_array_, index, is_leaf=lambda a: isinstance(a, Array))
        if not isinstance(value, Variable):
            raise ValueError('Must be instance of Variable.')
        super().__init__(value.value, batch_axis=value.batch_axis)
        self._value = value

    def __repr__(self) -> str:
        print_code = repr(self._value)
        prefix = f'{self.__class__.__name__}'
        blank = " " * (len(prefix) + 1)
        lines = print_code.split("\n")
        lines[0] = prefix + "(" + lines[0]
        for i in range(1, len(lines)):
            lines[i] = blank + lines[i]
        lines[-1] += ","
        lines.append(blank + f'index={self.index})')
        print_code = "\n".join(lines)
        return print_code

    @property
    def value(self):
        return self._value[self.index]

    @value.setter
    def value(self, v):
        int_shape = self.shape
        if self.batch_axis is None:
            ext_shape = v.shape
        else:
            ext_shape = v.shape[:self.batch_axis] + v.shape[self.batch_axis + 1:]
            int_shape = int_shape[:self.batch_axis] + int_shape[self.batch_axis + 1:]
        if ext_shape != int_shape:
            error = f"The shape of the original data is {self.shape}, while we got {v.shape}"
            if self.batch_axis is None:
                error += '. Do you forget to set "batch_axis" when initialize this variable?'
            else:
                error += f' with batch_axis={self.batch_axis}.'
            raise MathError(error)
        if v.dtype != self._value.dtype:
            raise MathError(f"The dtype of the original data is {self._value.dtype}, "
                            f"while we got {v.dtype}.")
        self._value[self.index] = v.value if isinstance(v, Array) else v


@register_pytree_node_class
class VarList(list):
    """A sequence of :py:class:`~.Variable`, which is compatible with
    :py:func:`.vars()` operation in a :py:class:`~.BrainPyObject`.

    Actually, :py:class:`~.VarList` is a python list.

    :py:class:`~.VarList` is specifically designed to store Variable instances.

    """

    def __init__(self, seq=()):
        super().__init__()
        self.extend(seq)

    def append(self, element) -> 'VarList':
        if not isinstance(element, Variable):
            raise TypeError(f'element must be an instance of {Variable.__name__}.')
        super().append(element)
        return self

    def extend(self, iterable) -> 'VarList':
        for element in iterable:
            self.append(element)
        return self

    def __setitem__(self, key, value) -> 'VarList':
        """Override the item setting.

        This function ensures that the Variable appended in the :py:class:`~.VarList` will not be overridden,
        and only the value can be changed for each element.

        >>> import brainpy.math as bm
        >>> l = bm.var_list([bm.Variable(1), bm.Variable(2)])
        >>> print(id(l[0]), id(l[1]))
        2077748389472 2077748389552
        >>> l[1] = bm.random.random(2)
        >>> l[0] = bm.random.random(1)
        >>> print(id(l[0]), id(l[1]))  # still the original Variable instances
        2077748389472 2077748389552
        """
        if isinstance(key, int):
            self[key].value = value
        else:
            super().__setitem__(key, value)
        return self

    def tree_flatten(self):
        return tuple(self), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children)


var_list = VarList


@register_pytree_node_class
class VarDict(dict):
    """A dictionary of :py:class:`~.Variable`, which is compatible with
    :py:func:`.vars()` operation in a :py:class:`~.BrainPyObject`.

    Actually, :py:class:`~.VarDict` is a python dict.

    :py:class:`~.VarDict` is specifically designed to store Variable instances.

    """

    def _check_elem(self, elem):
        if not isinstance(elem, Variable):
            raise TypeError(f'Element should be {Variable.__name__}, but got {type(elem)}.')
        return elem

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.update(*args, **kwargs)

    def update(self, *args, **kwargs) -> 'VarDict':
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
            elif isinstance(arg, tuple):
                assert len(arg) == 2
                self[arg[0]] = args[1]
        for k, v in kwargs.items():
            self[k] = v
        return self

    def __setitem__(self, key, value) -> 'VarDict':
        """Override the item setting.

        This function ensures that the Variable appended in the :py:class:`~.VarList` will not be overridden.

        >>> import brainpy.math as bm
        >>> d = bm.var_dict({'a': bm.Variable(1), 'b': bm.Variable(2)})
        >>> print(id(d['a']), id(d['b']))
        2077667833504 2077748488176
        >>> d['b'] = bm.random.random(2)
        >>> d['a'] = bm.random.random(1)
        >>> print(id(d['a']), id(d['b']))  # still the original Variable instances
        2077667833504 2077748488176
        """
        if key in self:
            self[key].value = value
        else:
            super().__setitem__(key, self._check_elem(value))
        return self

    def tree_flatten(self):
        return tuple(self.values()), tuple(self.keys())

    @classmethod
    def tree_unflatten(cls, keys, values):
        return cls(jax.util.safe_zip(keys, values))


var_dict = VarDict
