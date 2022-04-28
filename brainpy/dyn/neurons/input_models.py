# -*- coding: utf-8 -*-

import warnings
from typing import Union

import jax.numpy as jnp

import brainpy.math as bm
from brainpy.dyn.base import NeuGroup
from brainpy.errors import ModelBuildError
from brainpy.initialize import Initializer, init_param
from brainpy.types import Shape

__all__ = [
  'SpikeTimeInput',
  'PoissonInput',
  'SpikeTimeGroup',
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
      times,
      indices,
      need_sort: bool = True,
      name: str = None
  ):
    super(SpikeTimeGroup, self).__init__(size=size, name=name)

    # parameters
    if len(indices) != len(times):
      raise ModelBuildError(f'The length of "indices" and "times" must be the same. '
                            f'However, we got {len(indices)} != {len(times)}.')
    self.num_times = len(times)

    # data about times and indices
    self.times = bm.asarray(times, dtype=bm.float_)
    self.indices = bm.asarray(indices, dtype=bm.int_)

    # variables
    self.i = bm.Variable(bm.zeros(1, dtype=bm.int_))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    if need_sort:
      sort_idx = bm.argsort(self.times)
      self.indices.value = self.indices[sort_idx]
      self.times.value = self.times[sort_idx]

    # functions
    def cond_fun(t):
      return bm.logical_and(self.i[0] < self.num_times, t >= self.times[self.i[0]])

    def body_fun(t):
      self.spike[self.indices[self.i[0]]] = True
      self.i[0] += 1

    self._run = bm.make_while(cond_fun, body_fun, dyn_vars=self.vars())

  def reset(self):
    self.i[0] = 1
    self.spike[:] = False

  def update(self, t, _i, **kwargs):
    self.spike[:] = False
    self._run(t)


def SpikeTimeInput(*args, **kwargs):
  """Spike Time Input.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.SpikeTimeGroup" instead.

  Returns
  -------
  group: NeuGroup
    The neural group.
  """
  warnings.warn('Please use "brainpy.dyn.SpikeTimeGroup" instead. '
                '"brainpy.dyn.SpikeTimeInput" is deprecated since '
                'version 2.1.5', DeprecationWarning)
  return SpikeTimeGroup(*args, **kwargs)


class PoissonGroup(NeuGroup):
  """Poisson Neuron Group.
  """

  def __init__(
      self,
      size: Shape,
      freqs: Union[float, jnp.ndarray, bm.JaxArray, Initializer],
      seed: int = None,
      name: str = None
  ):
    super(PoissonGroup, self).__init__(size=size, name=name)

    # parameters
    self.seed = seed
    self.freqs = init_param(freqs, self.num, allow_none=False)
    self.dt = bm.get_dt() / 1000.
    self.size = (size,) if isinstance(size, int) else tuple(size)

    # variables
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.rng = bm.random.RandomState(seed=seed)

  def update(self, t, _i):
    self.spike.update(self.rng.random(self.num) <= self.freqs * self.dt)

  def reset(self):
    self.spike[:] = False
    self.rng.seed(self.seed)


def PoissonInput(*args, **kwargs):
  """Poisson Group Input.

  .. deprecated:: 2.1.0
     Please use "brainpy.dyn.PoissonGroup" instead.

  Returns
  -------
  poisson_group: NeuGroup
    The poisson neural group.
  """
  warnings.warn('Please use "brainpy.dyn.PoissonGroup" instead. '
                '"brainpy.dyn.PoissonInput" is deprecated since '
                'version 2.1.5', DeprecationWarning)
  return PoissonGroup(*args, **kwargs)
