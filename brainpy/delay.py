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
"""
Delay variable.
"""

import math
import numbers
from typing import Union, Dict, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from brainpy import check, math as bm
from brainpy.check import jit_error
from brainpy.context import share
from brainpy.dynsys import DynamicalSystem
from brainpy.initialize import variable_
from brainpy.math.delayvars import ROTATE_UPDATE, CONCAT_UPDATE
from brainpy.mixin import ParamDesc, ReturnInfo, JointType, SupportAutoDelay

__all__ = [
    'Delay',
    'VarDelay',
    'DataDelay',
    'DelayAccess',
]

delay_identifier = '_*_delay_of_'


def _get_delay(delay_time, delay_step):
    if delay_time is None:
        if delay_step is None:
            return None, None
        else:
            assert isinstance(delay_step, int), '"delay_step" should be an integer.'
            delay_time = delay_step * bm.get_dt()
    else:
        assert delay_step is None, '"delay_step" should be None if "delay_time" is given.'
        assert isinstance(delay_time, (int, float))
        delay_step = math.ceil(delay_time / bm.get_dt())
    return delay_time, delay_step


class Delay(DynamicalSystem, ParamDesc):
    """Base class for delay variables.

    Args:
      time: The delay time.
      init: The initial delay data.
      method: The delay method. Can be ``rotation`` and ``concat``.
      name: The delay name.
      mode: The computing mode.
    """

    max_time: float
    max_length: int
    data: Optional[bm.Variable]

    def __init__(
        self,
        # delay time
        time: Optional[Union[int, float]] = None,

        # delay init
        init: Optional[Union[numbers.Number, bm.Array, jax.Array, Callable]] = None,

        # delay method
        method: Optional[str] = None,

        # others
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,
    ):
        super().__init__(name=name, mode=mode)

        # delay method
        if method is None:
            if self.mode.is_one_of(bm.NonBatchingMode, bm.BatchingMode):
                method = ROTATE_UPDATE
            elif self.mode.is_a(bm.TrainingMode):
                method = CONCAT_UPDATE
            else:
                method = ROTATE_UPDATE
        assert method in [ROTATE_UPDATE, CONCAT_UPDATE]
        self.method = method

        # delay length
        if time is None:
            length = 0
            time = 0.
        elif isinstance(time, (int, float)):
            length = int(time / bm.get_dt())
        else:
            raise TypeError('time must be a int or float or None.')
        assert isinstance(length, int)
        self.max_length = length
        self.max_time = time

        # delay data
        if init is not None:
            assert isinstance(init, (numbers.Number, bm.Array, jax.Array, Callable))
        self._init = init

        # other info
        self._registered_entries = dict()

    def register_entry(
        self,
        entry: str,
        delay_time: Optional[Union[float, bm.Array, Callable]] = None,
        delay_step: Optional[int] = None
    ) -> 'Delay':
        """Register an entry to access the data.

        Args:
          entry: str. The entry to access the delay data.
          delay_time: The delay time of the entry (can be a float).
          delay_step: The delay step of the entry (must be an int). ``delay_step = delay_time / dt``.

        Returns:
          Return the self.
        """
        raise NotImplementedError

    def at(self, entry: str, *indices) -> bm.Array:
        """Get the data at the given entry.

        Args:
          entry: str. The entry to access the data.
          *indices: The slicing indices.

        Returns:
          The data.
        """
        raise NotImplementedError

    def retrieve(self, delay_step, *indices):
        """Retrieve the delay data according to the delay length.

        Parameters::

        delay_step: int, ArrayType
          The delay length used to retrieve the data.
        """
        raise NotImplementedError()


def _check_target_sharding(sharding, ndim, mode: bm.Mode):
    if sharding is not None:
        if len(sharding) == ndim:
            sharding = list(sharding)
        elif len(sharding) + 1 == ndim and mode.is_child_of(bm.BatchingMode):
            sharding = list(sharding)
            sharding.insert(0, bm.sharding.BATCH_AXIS)
        else:
            raise ValueError('sharding axis names do not match the target dimension. ')
    return sharding


class VarDelay(Delay):
    """Generate Delays for the given :py:class:`~.Variable` instance.

    The data in this delay variable is arranged as::

         delay = 0             [ data
         delay = 1               data
         delay = 2               data
         ...                     ....
         ...                     ....
         delay = length-1        data
         delay = length          data ]

    Args:
      target: Variable. The delay target.
      time: int, float. The delay time.
      init: Any. The delay data. It can be a Python number, like float, int, boolean values.
        It can also be arrays. Or a callable function or instance of ``Connector``.
        Note that ``initial_delay_data`` should be arranged as the following way::

           delay = 1             [ data
           delay = 2               data
           ...                     ....
           ...                     ....
           delay = length-1        data
           delay = length          data ]
      entries: optional, dict. The delay access entries.
      name: str. The delay name.
      method: str. The method used for updating delay. Default None.
      mode: Mode. The computing mode. Default None.

    """

    not_desc_params = ('time', 'entries', 'name')

    def __init__(
        self,

        # delay target
        target: bm.Variable,

        # delay time
        time: Optional[Union[int, float]] = None,

        # delay init
        init: Optional[Union[numbers.Number, bm.Array, jax.Array, Callable]] = None,

        # delay access entry
        entries: Optional[Dict] = None,

        # delay method
        method: Optional[str] = None,

        # others
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,
    ):
        super().__init__(time=time, init=init, method=method, name=name, mode=mode)

        # check
        if not isinstance(target, bm.Variable):
            raise ValueError(f'Must be an instance of brainpy.math.Variable. But we got {type(target)}')

        if self.mode.is_child_of(bm.BatchingMode):
            assert target.batch_axis is not None

        # sharding
        sharding = None
        if target.axis_names is not None:
            sharding = list(target.axis_names)
            sharding.insert(0, bm.sharding.TIME_AXIS)
            sharding = tuple(sharding)
        self.sharding = bm.sharding.get_sharding(sharding)

        # target
        self.target = target

        # delay data
        self._init = init
        if self.max_length > 0:
            self._init_data(self.max_length)
        else:
            self.data = None

        # other info
        if entries is not None:
            for entry, delay_time in entries.items():
                self.register_entry(entry, delay_time)

    def register_entry(
        self,
        entry: str,
        delay_time: Optional[Union[int, float]] = None,
        delay_step: Optional[int] = None,
    ) -> 'Delay':
        """Register an entry to access the data.

        Args:
          entry: str. The entry to access the delay data.
          delay_time: The delay time of the entry (can be a float).
          delay_step: The delay step of the entry (must be an int). ``delat_step = delay_time / dt``.

        Returns:
          Return the self.
        """
        if entry in self._registered_entries:
            raise KeyError(f'Entry {entry} has been registered. '
                           f'The existing delay for the key {entry} is {self._registered_entries[entry]}. '
                           f'The new delay for the key {entry} is {delay_time}. '
                           f'You can use another key. ')

        if isinstance(delay_time, (np.ndarray, jax.Array)):
            assert delay_time.size == 1 and delay_time.ndim == 0
            delay_time = delay_time.item()

        _, delay_step = _get_delay(delay_time, delay_step)

        # delay variable
        if delay_step is not None:
            if self.max_length < delay_step:
                self._init_data(delay_step)
                self.max_length = delay_step
                self.max_time = delay_time
        self._registered_entries[entry] = delay_step
        return self

    def at(self, entry: str, *indices) -> bm.Array:
        """Get the data at the given entry.

        Args:
          entry: str. The entry to access the data.
          *indices: The slicing indices. Not include the slice at the batch dimension.

        Returns:
          The data.
        """
        assert isinstance(entry, str), 'entry should be a string for describing the '
        if entry not in self._registered_entries:
            raise KeyError(f'Does not find delay entry "{entry}".')
        delay_step = self._registered_entries[entry]
        if isinstance(self.mode, bm.BatchingMode) and len(indices) > self.target.batch_axis:
            indices = list(indices)
            indices.insert(self.target.batch_axis, slice(None, None, None))
            indices = tuple(indices)

        if delay_step is None or delay_step == 0.:
            if len(indices):
                return self.target.value[indices]
            else:
                return self.target.value
        else:
            return self.retrieve(delay_step, *indices)

    @property
    def delay_target_shape(self):
        """The data shape of the delay target."""
        return self.target.shape

    def __repr__(self):
        name = self.__class__.__name__
        return f'{name}(step={self.max_length}, shape={self.delay_target_shape}, method={self.method})'

    def _check_delay(self, delay_len):
        raise ValueError(f'The request delay length should be less than the '
                         f'maximum delay {self.max_length}. '
                         f'But we got {delay_len}')

    def retrieve(self, delay_step, *indices):
        """Retrieve the delay data according to the delay length.

        Parameters::

        delay_step: int, Array
          The delay length used to retrieve the data.
        """
        assert self.data is not None
        assert delay_step is not None
        if check.is_checking():
            jit_error(delay_step > self.max_length, self._check_delay, delay_step)

        if self.method == ROTATE_UPDATE:
            i = share.load('i')
            delay_idx = bm.as_jax((delay_step - i - 1) % self.max_length, dtype=jnp.int32)
            delay_idx = jax.lax.stop_gradient(delay_idx)

        elif self.method == CONCAT_UPDATE:
            delay_idx = delay_step - 1

        else:
            raise ValueError(f'Unknown updating method "{self.method}"')

        # the delay index
        if hasattr(delay_idx, 'dtype') and not jnp.issubdtype(delay_idx.dtype, jnp.integer):
            raise ValueError(f'"delay_len" must be integer, but we got {delay_idx}')
        indices = (delay_idx,) + indices

        # the delay data
        return self.data.value[indices]

    def update(
        self,
        latest_value: Optional[Union[bm.Array, jax.Array]] = None
    ) -> None:
        """Update delay variable with the new data.
        """
        if self.data is not None:
            # jax.debug.print('last value == target value {} ', jnp.allclose(latest_value, self.target.value))

            # get the latest target value
            if latest_value is None:
                latest_value = self.target.value

            # update the delay data at the rotation index
            if self.method == ROTATE_UPDATE:
                i = share.load('i')
                idx = bm.as_jax(-i % self.max_length, dtype=jnp.int32)
                self.data[jax.lax.stop_gradient(idx)] = latest_value

            # update the delay data at the first position
            elif self.method == CONCAT_UPDATE:
                if self.max_length > 1:
                    latest_value = bm.expand_dims(latest_value, 0)
                    self.data.value = bm.concat([latest_value, self.data[:-1]], axis=0)
                else:
                    self.data[0] = latest_value

            else:
                raise ValueError(f'Unknown updating method "{self.method}"')

    def reset_state(self, batch_size: int = None, **kwargs):
        """Reset the delay data.
        """
        # initialize delay data
        if self.data is not None:
            self._init_data(self.max_length, batch_size)

    def _init_data(self, length: int, batch_size: int = None):
        if batch_size is not None:
            if self.target.batch_size != batch_size:
                raise ValueError(f'The batch sizes of delay variable and target variable differ '
                                 f'({self.target.batch_size} != {batch_size}). '
                                 'Please reset the target variable first, because delay data '
                                 'depends on the target variable. ')

        if self.target.batch_axis is None:
            batch_axis = None
        else:
            batch_axis = self.target.batch_axis + 1

        if self.sharding is None:
            f = jnp.zeros
        else:
            f = jax.jit(jnp.zeros, static_argnums=0, static_argnames='dtype', out_shardings=self.sharding)

        data = f((length,) + self.target.shape, dtype=self.target.dtype)
        if self.data is None:
            self.data = bm.Variable(data, batch_axis=batch_axis)
        else:
            self.data._value = data
        # update delay data
        if isinstance(self._init, (bm.Array, jax.Array, numbers.Number)):
            self.data[:] = self._init
        elif callable(self._init):
            self.data[:] = self._init((length,) + self.target.shape, dtype=self.target.dtype)
        else:
            assert self._init is None, f'init should be Array, Callable, or None. but got {self._init}'


class DataDelay(VarDelay):
    not_desc_params = ('time', 'entries', 'name')

    def __init__(
        self,

        # delay target
        data: bm.Variable,
        data_init: Union[Callable, bm.Array, jax.Array],

        # delay time
        time: Optional[Union[int, float]] = None,

        # delay init
        init: Optional[Union[numbers.Number, bm.Array, jax.Array, Callable]] = None,

        # delay access entry
        entries: Optional[Dict] = None,

        # delay method
        method: Optional[str] = None,

        # others
        name: Optional[str] = None,
        mode: Optional[bm.Mode] = None,
    ):
        self.target_init = data_init
        super().__init__(target=data,
                         time=time,
                         init=init,
                         entries=entries,
                         method=method,
                         name=name,
                         mode=mode)

    def reset_state(self, batch_size: int = None, **kwargs):
        """Reset the delay data.
        """
        self.target.value = variable_(self.target_init, self.target.size_without_batch, batch_size)
        if self.data is not None:
            self._init_data(self.max_length, batch_size)

    def update(
        self,
        latest_value: Union[bm.Array, jax.Array]
    ) -> None:
        """Update delay variable with the new data.
        """
        self.target.value = latest_value
        super().update(latest_value)


class DelayAccess(DynamicalSystem):
    def __init__(
        self,
        delay: Delay,
        time: Union[None, int, float],
        *indices,
        delay_entry: str = None
    ):
        super().__init__(mode=delay.mode)
        self.refs = {'delay': delay}
        assert isinstance(delay, Delay)
        self._delay_entry = delay_entry or self.name
        delay.register_entry(self._delay_entry, time)
        self.indices = indices

    def update(self):
        return self.refs['delay'].at(self._delay_entry, *self.indices)

    def reset_state(self, *args, **kwargs):
        pass


def init_delay_by_return(info: Union[bm.Variable, ReturnInfo], initial_delay_data=None) -> Delay:
    """Initialize a delay class by the return info (usually is created by ``.return_info()`` function).

    Args:
      info: the return information.
      initial_delay_data: The initial delay data.

    Returns:
      The decay instance.
    """
    if isinstance(info, bm.Variable):
        return VarDelay(info, init=initial_delay_data)

    elif isinstance(info, ReturnInfo):
        # batch size
        if isinstance(info.batch_or_mode, int):
            shape = (info.batch_or_mode,) + tuple(info.size)
            batch_axis = 0
        elif isinstance(info.batch_or_mode, bm.NonBatchingMode):
            shape = tuple(info.size)
            batch_axis = None
        elif isinstance(info.batch_or_mode, bm.BatchingMode):
            shape = (info.batch_or_mode.batch_size,) + tuple(info.size)
            batch_axis = 0
        else:
            shape = tuple(info.size)
            batch_axis = None

        # init
        if isinstance(info.data, Callable):
            init = info.data(shape)
        elif isinstance(info.data, (bm.Array, jax.Array)):
            init = info.data
        else:
            raise TypeError
        assert init.shape == shape

        # axis names
        if info.axis_names is not None:
            assert init.ndim == len(info.axis_names)

        # variable
        target = bm.Variable(init, batch_axis=batch_axis, axis_names=info.axis_names)
        return DataDelay(target, data_init=info.data, init=initial_delay_data)
    else:
        raise TypeError


def register_delay_by_return(target: JointType[DynamicalSystem, SupportAutoDelay]):
    """Register delay class for the given target.

    Args:
      target: The target class to register delay.

    Returns:
      The delay registered for the given target.
    """
    if not target.has_aft_update(delay_identifier):
        delay_ins = init_delay_by_return(target.return_info())
        target.add_aft_update(delay_identifier, delay_ins)
    delay_cls = target.get_aft_update(delay_identifier)
    return delay_cls
