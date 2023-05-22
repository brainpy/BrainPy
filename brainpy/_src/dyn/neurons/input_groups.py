# -*- coding: utf-8 -*-

from typing import Union, Sequence

import jax.numpy as jnp
from brainpy._src.context import share
import brainpy.math as bm
from brainpy._src.dynsys import NeuGroupNS
from brainpy._src.initialize import Initializer, parameter, variable_
from brainpy.types import Shape, ArrayType

__all__ = [
  'InputGroup',
  'OutputGroup',
  'SpikeTimeGroup',
  'PoissonGroup',
]


class InputGroup(NeuGroupNS):
  """Input neuron group for place holder.

  Parameters
  ----------
  size: int, tuple of int
  keep_size: bool
  mode: Mode
  name: str
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      mode: bm.Mode = None,
      name: str = None,
  ):
    super(InputGroup, self).__init__(name=name,
                                     size=size,
                                     keep_size=keep_size,
                                     mode=mode)
    self.spike = None

  def update(self, x):
    return x

  def reset_state(self, batch_size=None):
    pass


class OutputGroup(NeuGroupNS):
  """Output neuron group for place holder.

  Parameters
  ----------
  size: int, tuple of int
  keep_size: bool
  mode: Mode
  name: str
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      mode: bm.Mode = None,
      name: str = None,
  ):
    super(OutputGroup, self).__init__(name=name,
                                      size=size,
                                      keep_size=keep_size,
                                      mode=mode)
    self.spike = None

  def update(self, x):
    return x

  def reset_state(self, batch_size=None):
    pass


class SpikeTimeGroup(NeuGroupNS):
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
      size: Shape,
      times: Union[Sequence, ArrayType],
      indices: Union[Sequence, ArrayType],
      need_sort: bool = True,
      keep_size: bool = False,
      mode: bm.Mode = None,
      name: str = None
  ):
    super(SpikeTimeGroup, self).__init__(size=size,
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
    self.spike = variable_(lambda s: jnp.zeros(s, dtype=bool), self.varshape, batch_size)

  def update(self):
    self.spike.value = bm.zeros_like(self.spike)
    bm.while_loop(self._cond_fun, self._body_fun, share.load('t'))
    return self.spike.value

  # functions
  def _cond_fun(self, t):
    i = self.i.value
    return bm.logical_and(i < self.num_times, t >= self.times[i])

  def _body_fun(self, t):
    i = self.i.value
    if isinstance(self.mode, bm.BatchingMode):
      self.spike[:, self.indices[i]] = True
    else:
      self.spike[self.indices[i]] = True
    self.i += 1


class PoissonGroup(NeuGroupNS):
  """Poisson Neuron Group.
  """

  def __init__(
      self,
      size: Shape,
      freqs: Union[int, float, jnp.ndarray, bm.Array, Initializer],
      seed: int = None,
      keep_size: bool = False,
      mode: bm.Mode = None,
      name: str = None
  ):
    super(PoissonGroup, self).__init__(size=size,
                                       name=name,
                                       keep_size=keep_size,
                                       mode=mode)

    # parameters
    self.keep_size = keep_size
    self.seed = seed
    self.freqs = parameter(freqs, self.num, allow_none=False)

    # variables
    self.rng = bm.random.default_rng(seed)
    self.reset_state(self.mode)

  def update(self):
    spikes = self.rng.rand_like(self.spike) <= (self.freqs * share.dt / 1000.)
    self.spike.value = spikes
    return spikes

  def reset(self, batch_size=None):
    self.rng.value = bm.random.default_rng(self.seed)
    self.reset_state(batch_size)

  def reset_state(self, batch_size=None):
    self.spike = variable_(lambda s: jnp.zeros(s, dtype=bool), self.varshape, batch_size)

