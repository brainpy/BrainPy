# -*- coding: utf-8 -*-
import warnings
from functools import partial
from typing import Union, Sequence, Any, Optional, Callable

import jax
import jax.numpy as jnp

from brainpy import math as bm
from brainpy._src.context import share
from brainpy._src.dyn.utils import get_spk_type
from brainpy._src.dyn.base import NeuDyn
from brainpy._src.initialize import parameter, variable_
from brainpy._src.mixin import ReturnInfo
from brainpy.types import Shape, ArrayType

__all__ = [
  'InputGroup',
  'OutputGroup',
  'SpikeTimeGroup',
  'PoissonGroup',
]


class InputGroup(NeuDyn):
  """Input neuron group for place holder.

  Args:
    size: int, tuple of int
    keep_size: bool
    mode: Mode
    name: str
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      sharding: Any = None,
      keep_size: bool = False,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name,
                     sharding=sharding,
                     size=size,
                     keep_size=keep_size,
                     mode=mode)

  def update(self, x):
    return x

  def return_info(self):
    return ReturnInfo(self.varshape, self.sharding, self.mode, bm.zeros)

  def reset_state(self, batch_size=None):
    pass


class OutputGroup(NeuDyn):
  """Output neuron group for place holder.

  Args:
    size: int, tuple of int
    keep_size: bool
    mode: Mode
    name: str
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      sharding: Any = None,
      keep_size: bool = False,
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None,
  ):
    super().__init__(name=name,
                     sharding=sharding,
                     size=size,
                     keep_size=keep_size,
                     mode=mode)

  def update(self, x):
    return x

  def return_info(self):
    return ReturnInfo(self.varshape, self.sharding, self.mode, bm.zeros)

  def reset_state(self, batch_size=None):
    pass


class SpikeTimeGroup(NeuDyn):
  """The input neuron group characterized by spikes emitting at given times.

  >>> # Get 2 neurons, firing spikes at 10 ms and 20 ms.
  >>> SpikeTimeGroup(2, times=[10, 20])
  >>> # or
  >>> # Get 2 neurons, the neuron 0 fires spikes at 10 ms and 20 ms.
  >>> SpikeTimeGroup(2, times=[10, 20], indices=[0, 0])
  >>> # or
  >>> # Get 2 neurons, neuron 0 fires at 10 ms and 30 ms, neuron 1 fires at 20 ms.
  >>> SpikeTimeGroup(2, times=[10, 20, 30], indices=[0, 1, 0])
  >>> # or
  >>> # Get 2 neurons; at 10 ms, neuron 0 fires; at 20 ms, neuron 0 and 1 fire;
  >>> # at 30 ms, neuron 1 fires.
  >>> SpikeTimeGroup(2, times=[10, 20, 20, 30], indices=[0, 0, 1, 1])

  Parameters
  ----------
  size : int, tuple, list
      The neuron group geometry.
  indices : list, tuple, ArrayType
      The neuron indices at each time point to emit spikes.
  times : list, tuple, ArrayType
      The time points which generate the spikes.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      indices: Union[Sequence, ArrayType],
      times: Union[Sequence, ArrayType],
      spk_type: Optional[type] = None,
      name: Optional[str] = None,
      sharding: Optional[Sequence[str]] = None,
      keep_size: bool = False,
      mode: Optional[bm.Mode] = None,
      need_sort: bool = True,
  ):
    super().__init__(size=size,
                     sharding=sharding,
                     name=name,
                     keep_size=keep_size,
                     mode=mode)

    # parameters
    if keep_size:
      raise NotImplementedError(f'Do not support keep_size=True in {self.__class__.__name__}')
    if len(indices) != len(times):
      raise ValueError(f'The length of "indices" and "times" must be the same. '
                       f'However, we got {len(indices)} != {len(times)}.')
    self.num_times = len(times)
    self.spk_type = get_spk_type(spk_type, self.mode)

    # data about times and indices
    self.times = bm.asarray(times)
    self.indices = bm.asarray(indices, dtype=bm.int_)
    if need_sort:
      sort_idx = bm.argsort(self.times)
      self.indices.value = self.indices[sort_idx]
      self.times.value = self.times[sort_idx]

    # variables
    self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    self.i = bm.Variable(bm.asarray(0))
    self.spike = variable_(partial(jnp.zeros, dtype=self.spk_type),
                           self.varshape,
                           batch_size,
                           axis_names=self.sharding,
                           batch_axis_name=bm.sharding.BATCH_AXIS)

  def update(self):
    # self.spike.value = bm.sharding.partition(bm.zeros_like(self.spike), self.spike.sharding)
    self.spike.value = bm.zeros_like(self.spike)
    bm.while_loop(self._body_fun, self._cond_fun, ())
    return self.spike.value

  def return_info(self):
    return self.spike

  # functions
  def _cond_fun(self):
    i = self.i.value
    return bm.logical_and(i < self.num_times, share['t'] >= self.times[i])

  def _body_fun(self):
    i = self.i.value
    if isinstance(self.mode, bm.BatchingMode):
      self.spike[:, self.indices[i]] = True
    else:
      self.spike[self.indices[i]] = True
    self.i += 1


class PoissonGroup(NeuDyn):
  """Poisson Neuron Group.
  """

  def __init__(
      self,
      size: Shape,
      freqs: Union[int, float, jax.Array, bm.Array, Callable],
      keep_size: bool = False,
      sharding: Optional[Sequence[str]] = None,
      spk_type: Optional[type] = None,
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
      seed=None,
  ):
    super().__init__(size=size,
                     sharding=sharding,
                     name=name,
                     keep_size=keep_size,
                     mode=mode)

    if seed is not None:
      warnings.warn('')

    # parameters
    self.freqs = parameter(freqs, self.num, allow_none=False)
    self.spk_type = get_spk_type(spk_type, self.mode)

    # variables
    self.reset_state(self.mode)

  def update(self):
    spikes = bm.random.rand_like(self.spike) <= (self.freqs * share['dt'] / 1000.)
    spikes = bm.asarray(spikes, dtype=self.spk_type)
    # spikes = bm.sharding.partition(spikes, self.spike.sharding)
    self.spike.value = spikes
    return spikes

  def return_info(self):
    return self.spike

  def reset_state(self, batch_size=None):
    self.spike = variable_(partial(jnp.zeros, dtype=self.spk_type),
                           self.varshape,
                           batch_size,
                           axis_names=self.sharding,
                           batch_axis_name=bm.sharding.BATCH_AXIS)
