# -*- coding: utf-8 -*-

from typing import Union, Sequence

import jax.numpy as jnp

import brainpy.math as bm
from brainpy.dyn.base import NeuGroup
from brainpy.errors import ModelBuildError
from brainpy.initialize import Initializer, init_param
from brainpy.types import Shape, Tensor

__all__ = [
  'SpikeTimeInput',
  'SpikeTimeGroup',

  'PoissonInput',
  'PoissonGroup',
]


class SpikeTimeGroup(NeuGroup):
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
  indices : list, tuple, np.ndarray, JaxArray, jax.numpy.ndarray
      The neuron indices at each time point to emit spikes.
  times : list, tuple, np.ndarray, JaxArray, jax.numpy.ndarray
      The time points which generate the spikes.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(
      self,
      size: Shape,
      times: Union[Sequence, Tensor],
      indices: Union[Sequence, Tensor],
      need_sort: bool = True,
      keep_size: bool = False,
      name: str = None
  ):
    super(SpikeTimeGroup, self).__init__(size=size, name=name)

    # parameters
    self.keep_size = keep_size
    if keep_size:
      raise NotImplementedError(f'Do not support keep_size=True in {self.__class__.__name__}')
    if len(indices) != len(times):
      raise ModelBuildError(f'The length of "indices" and "times" must be the same. '
                            f'However, we got {len(indices)} != {len(times)}.')
    self.num_times = len(times)

    # data about times and indices
    self.times = bm.asarray(times)
    self.indices = bm.asarray(indices)

    # variables
    self.i = bm.Variable(bm.zeros(1))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    if need_sort:
      sort_idx = bm.argsort(self.times)
      self.indices.value = self.indices[sort_idx]
      self.times.value = self.times[sort_idx]

    # functions
    def cond_fun(t):
      i = self.i[0]
      return bm.logical_and(i < self.num_times, t >= self.times[i])

    def body_fun(t):
      self.spike[self.indices[self.i[0]]] = True
      self.i += 1

    self._run = bm.make_while(cond_fun, body_fun, dyn_vars=self.vars())

  def reset(self):
    self.i[0] = 1
    self.spike[:] = False

  def update(self, t, dt):
    self.spike[:] = False
    self._run(t)


class SpikeTimeInput(SpikeTimeGroup):
  """Spike Time Input.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.SpikeTimeGroup" instead.

  Returns
  -------
  group: NeuGroup
    The neural group.
  """

  def __init__(self, *args, **kwargs):
    raise ValueError('Please use "brainpy.dyn.SpikeTimeGroup" instead. '
                     '"brainpy.dyn.SpikeTimeInput" is deprecated since '
                     'version 2.1.5')


class PoissonGroup(NeuGroup):
  """Poisson Neuron Group.
  """

  def __init__(
      self,
      size: Shape,
      freqs: Union[float, jnp.ndarray, bm.JaxArray, Initializer],
      seed: int = None,
      keep_size: bool = False,
      name: str = None
  ):
    super(PoissonGroup, self).__init__(size=size, name=name)

    # parameters
    self.keep_size = keep_size
    self.seed = seed
    self.freqs = init_param(freqs, self.num, allow_none=False)

    # variables
    self.spike = bm.Variable(bm.zeros(self.size if keep_size else self.num, dtype=bool))
    self.rng = bm.random.RandomState(seed=seed)

  def update(self, t, dt):
    self.spike.update(self.rng.random(self.var_shape) <= (self.freqs * dt / 1000.))

  def reset(self):
    self.spike[:] = False
    self.rng.seed(self.seed)


class PoissonInput(PoissonGroup):
  """Poisson Group Input.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.PoissonGroup" instead.

  Returns
  -------
  poisson_group: NeuGroup
    The poisson neural group.
  """

  def __init__(self, *args, **kwargs):
    raise ValueError('Please use "brainpy.dyn.PoissonGroup" instead. '
                     '"brainpy.dyn.PoissonInput" is deprecated since '
                     'version 2.1.5')
