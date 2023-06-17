"""
Delay variable.
"""

import numbers
from dataclasses import dataclass
from typing import Union, Callable, Optional, Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import stop_gradient

from brainpy import check
from brainpy import math as bm, tools
from brainpy._src.context import share
from brainpy._src.dynsys import DynamicalSystemNS
from brainpy._src.math.delayvars import ROTATE_UPDATE, CONCAT_UPDATE
from brainpy._src.pnn.utils.axis_names import NEU_AXIS, TIME_AXIS
from brainpy.check import jit_error
from brainpy.types import ArrayType
from .mixin import ParamDesc

__all__ = [
  'TargetDelay',
  'DataDelay',
]


@dataclass(frozen=True)
class DelayDesc:
  time: Optional[Union[int, float]] = None
  dtype: Optional[type] = None
  init: Optional[Union[numbers.Number, ArrayType, Callable]] = None
  target: Optional[bm.Variable] = None


class Delay(DynamicalSystemNS, ParamDesc):
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
      if self.mode.is_parent_of(bm.NonBatchingMode):
        method = ROTATE_UPDATE
      elif self.mode.is_parent_of(bm.TrainingMode):
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
      delay_time: Optional[Union[float, bm.Array, Callable]],
  ) -> 'Delay':
    """Register an entry to access the data.

    Args:
      entry: str. The entry to access the delay data.
      delay_time: The delay time of the entry (can be a float).

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

    Parameters
    ----------
    delay_step: int, ArrayType
      The delay length used to retrieve the data.
    """
    raise NotImplementedError()


class TargetDelay(Delay):
  """Delay variable which has a fixed delay length.

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
    axis_names: sequence of str. The name for each axis.
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

  not_desc_params = ('time', 'entries')

  def __init__(
      self,

      # delay target
      target: bm.Variable,
      axis_names: Optional[Sequence[str]] = None,

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

    # target
    if not isinstance(target, bm.Variable):
      raise ValueError(f'Must be an instance of brainpy.math.Variable. But we got {type(target)}')

    if axis_names is not None:
      if len(axis_names) == target.ndim:
        axis_names = list(axis_names)
      elif len(axis_names) + 1 == target.ndim and isinstance(self.mode, bm.BatchingMode):
        axis_names = list(axis_names)
        axis_names.insert(0, NEU_AXIS)
      else:
        raise ValueError
    self.target_axis_names = axis_names
    if axis_names is not None:
      axis_names = list(axis_names)
      axis_names.insert(0, TIME_AXIS)
      self.data_axis_names = axis_names

    if self.mode.is_child_of(bm.BatchingMode):
      assert target.batch_axis is not None
    target = bm.sharding.partition_by_axname(target, self.target_axis_names)
    self.target = target

    # delay data
    self._init = init
    if self.max_length > 0:
      self._init_data(self.max_length)
    else:
      self.data = None

    # other info
    if entries is not None:
      for entry, value in entries.items():
        self.register_entry(entry, value)

  def register_entry(
      self,
      entry: str,
      delay_time: Optional[Union[float, bm.Array, Callable]],
  ) -> 'Delay':
    """Register an entry to access the data.

    Args:
      entry: str. The entry to access the delay data.
      delay_time: The delay time of the entry (can be a float).

    Returns:
      Return the self.
    """
    if entry in self._registered_entries:
      raise KeyError(f'Entry {entry} has been registered.')

    if delay_time is None:
      delay_step = None
      delay_time = 0.
    elif callable(delay_time):
      delay_time = bm.as_jax(delay_time(self.delay_target_shape))
      delay_step = jnp.asarray(delay_time / bm.get_dt(), dtype=bm.get_int())
    elif isinstance(delay_time, float):
      delay_step = int(delay_time / bm.get_dt())
    else:
      delay_step = jnp.asarray(bm.as_jax(delay_time) / bm.get_dt(), dtype=bm.get_int())

    # delay steps
    if delay_step is None:
      delay_type = 'none'
    elif isinstance(delay_step, int):
      delay_type = 'homo'
    elif isinstance(delay_step, (bm.Array, jax.Array, np.ndarray)):
      if delay_step.size == 1 and delay_step.ndim == 0:
        delay_type = 'homo'
      else:
        delay_type = 'heter'
        delay_step = bm.Array(delay_step)
    elif callable(delay_step):
      delay_step = delay_step(self.delay_target_shape)
      delay_type = 'heter'
    else:
      raise ValueError(f'Unknown "delay_steps" type {type(delay_step)}, only support '
                       f'integer, array of integers, callable function, brainpy.init.Initializer.')
    if delay_type == 'heter':
      if delay_step.dtype not in [jnp.int32, jnp.int64]:
        raise ValueError('Only support delay steps of int32, int64. If your '
                         'provide delay time length, please divide the "dt" '
                         'then provide us the number of delay steps.')
      if self.delay_target_shape[0] != delay_step.shape[0]:
        raise ValueError(f'Shape is mismatched: {self.delay_target_shape[0]} != {delay_step.shape[0]}')
    if delay_type == 'heter':
      max_delay_step = int(max(delay_step))
    elif delay_type == 'homo':
      max_delay_step = delay_step
    else:
      max_delay_step = None

    # delay variable
    if max_delay_step is not None:
      if self.max_length < max_delay_step:
        self._init_data(max_delay_step)
        self.max_length = max_delay_step
        self.max_time = delay_time
    self._registered_entries[entry] = delay_step
    return self

  def at(self, entry: str, *indices) -> bm.Array:
    """Get the data at the given entry.

    Args:
      entry: str. The entry to access the data.
      *indices: The slicing indices.

    Returns:
      The data.
    """
    assert isinstance(entry, str), 'entry should be a string for describing the '
    if entry not in self._registered_entries:
      raise KeyError(f'Does not find delay entry "{entry}".')
    delay_step = self._registered_entries[entry]
    if delay_step is None:
      return self.target.value
    else:
      if self.data is None:
        return self.target.value
      else:
        if isinstance(delay_step, slice):
          return self.retrieve(delay_step, *indices)
        elif np.ndim(delay_step) == 0:
          return self.retrieve(delay_step, *indices)
        else:
          if len(indices) == 0 and len(delay_step) == self.target.shape[0]:
            indices = (jnp.arange(delay_step.size),)
          return self.retrieve(delay_step, *indices)

  @property
  def delay_target_shape(self):
    """The data shape of the delay target."""
    return self.target.shape

  def __repr__(self):
    name = self.__class__.__name__
    return (f'{name}(num_delay_step={self.max_length}, '
            f'delay_target_shape={self.delay_target_shape}, '
            f'update_method={self.method})')

  def _check_delay(self, delay_len):
    raise ValueError(f'The request delay length should be less than the '
                     f'maximum delay {self.max_length}. '
                     f'But we got {delay_len}')

  def retrieve(self, delay_step, *indices):
    """Retrieve the delay data according to the delay length.

    Parameters
    ----------
    delay_step: int, ArrayType
      The delay length used to retrieve the data.
    """
    assert delay_step is not None
    if check.is_checking():
      jit_error(bm.any(delay_step > self.max_length), self._check_delay, delay_step)

    if self.method == ROTATE_UPDATE:
      i = share.load('i')
      delay_idx = (i + delay_step) % (self.max_length + 1)
      delay_idx = stop_gradient(delay_idx)

    elif self.method == CONCAT_UPDATE:
      delay_idx = delay_step

    else:
      raise ValueError(f'Unknown updating method "{self.method}"')

    # the delay index
    if hasattr(delay_idx, 'dtype') and not jnp.issubdtype(delay_idx.dtype, jnp.integer):
      raise ValueError(f'"delay_len" must be integer, but we got {delay_idx}')
    indices = (delay_idx,) + tuple(indices)

    # the delay data
    return self.data[indices]

  def update(
      self,
      latest_value: Optional[Union[bm.Array, jax.Array]] = None
  ) -> None:
    """Update delay variable with the new data.
    """
    if self.data is not None:
      # get the latest target value
      if latest_value is None:
        latest_value = self.target.value

      # update the delay data at the rotation index
      if self.method == ROTATE_UPDATE:
        i = share.load('i')
        idx = bm.as_jax((i - 1) % (self.max_length + 1))
        self.data[idx] = latest_value

      # update the delay data at the first position
      elif self.method == CONCAT_UPDATE:
        if self.max_length >= 2:
          self.data.value = bm.vstack([latest_value, self.data[1:]])
        else:
          self.data[0] = latest_value

  def reset_state(self, batch_size: int = None):
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

    f = jax.jit(jnp.zeros, static_argnums=0, static_argnames='dtype',
                out_shardings=bm.sharding.get_sharding(self.data_axis_names))
    data = f((length + 1,) + self.target.shape, dtype=self.target.dtype)
    self.data = bm.Variable(data, batch_axis=batch_axis)
    # update delay data
    self.data[0] = self.target.value
    if isinstance(self._init, (bm.Array, jax.Array, numbers.Number)):
      self.data[1:] = self._init
    elif callable(self._init):
      self.data[1:] = self._init((length,) + self.target.shape,
                                 dtype=self.target.dtype)


class DataDelay(TargetDelay):
  """Delay variable which has a fixed delay length.

  The data in this delay variable is arranged as::

       delay = 0             [ data
       delay = 1               data
       delay = 2               data
       ...                     ....
       ...                     ....
       delay = length-1        data
       delay = length          data ]

  Args:
    size: int, sequence of int. The delay target size.
    axis_names: sequence of str. The name for each axis.
    time: optional, int, float. The delay time. Default is None.
    dtype: type. The data type.
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

  not_desc_params = ('time', 'entries')

  def __init__(
      self,

      # delay info
      size: Union[int, Sequence[int]],
      axis_names: Optional[Sequence[str]] = None,
      dtype: Optional[type] = None,

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
    size = tools.to_size(size)
    mode = mode if mode is not None else bm.get_mode()
    if axis_names is not None:
      assert len(size) == len(axis_names)
    if isinstance(mode, bm.BatchingMode):
      batch_axis = 0
      size = (mode.batch_size,) + size
    else:
      batch_axis = None

    target = bm.Variable(bm.zeros(size, dtype=dtype), batch_axis=batch_axis)
    if init is None:
      pass
    elif isinstance(init, (bm.Array, jax.Array, numbers.Number)):
      target[:] = self._init
    elif callable(self._init):
      target[:] = self._init(size, dtype=dtype)
    else:
      raise ValueError

    super().__init__(target=target,
                     axis_names=axis_names,
                     time=time,
                     init=init,
                     entries=entries,
                     method=method,
                     name=name,
                     mode=mode)

  def update(
      self,
      latest_value: Union[bm.Array, jax.Array]
  ) -> None:
    """Update delay variable with the new data.
    """
    latest_value = bm.sharding.partition_by_axname(latest_value, self.target_axis_names)

    # get the latest target value
    self.target.value = latest_value

    if self.data is not None:
      # update the delay data at the rotation index
      if self.method == ROTATE_UPDATE:
        i = share.load('i')
        idx = bm.as_jax((i - 1) % (self.max_length + 1))
        self.data[idx] = latest_value

      # update the delay data at the first position
      elif self.method == CONCAT_UPDATE:
        if self.max_length >= 2:
          self.data.value = bm.vstack([latest_value, self.data[1:]])
        else:
          self.data[0] = latest_value
