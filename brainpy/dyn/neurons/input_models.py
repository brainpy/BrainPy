# -*- coding: utf-8 -*-

import jax.numpy as jnp

import brainpy.math as bm
from brainpy.dyn.base import NeuGroup
from brainpy.errors import ModelBuildError

__all__ = [
  'SpikeTimeInput',
  'PoissonInput',
  'SpikeTimeGroup',
  'PoissonGroup',
]


class SpikeTimeGroup(NeuGroup):
  """The input neuron group characterized by spikes emitting at given times.

  >>> # Get 2 neurons, firing spikes at 10 ms and 20 ms.
  >>> SpikeTimeInput(2, times=[10, 20])
  >>> # or
  >>> # Get 2 neurons, the neuron 0 fires spikes at 10 ms and 20 ms.
  >>> SpikeTimeInput(2, times=[10, 20], indices=[0, 0])
  >>> # or
  >>> # Get 2 neurons, neuron 0 fires at 10 ms and 30 ms, neuron 1 fires at 20 ms.
  >>> SpikeTimeInput(2, times=[10, 20, 30], indices=[0, 1, 0])
  >>> # or
  >>> # Get 2 neurons; at 10 ms, neuron 0 fires; at 20 ms, neuron 0 and 1 fire;
  >>> # at 30 ms, neuron 1 fires.
  >>> SpikeTimeInput(2, times=[10, 20, 20, 30], indices=[0, 0, 1, 1])

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

  def __init__(self, size, times, indices, need_sort=True, name=None):
    super(SpikeTimeGroup, self).__init__(size=size, name=name)

    # parameters
    if len(indices) != len(times):
      raise ModelBuildError(f'The length of "indices" and "times" must be the same. '
                            f'However, we got {len(indices)} != {len(times)}.')
    self.num_times = len(times)

    # data about times and indices
    self.i = bm.Variable(jnp.zeros(1, dtype=bm.int_))
    self.times = bm.Variable(jnp.asarray(times, dtype=bm.float_))
    self.indices = bm.Variable(jnp.asarray(indices, dtype=bm.int_))
    self.spike = bm.Variable(jnp.zeros(self.num, dtype=bool))
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

  def update(self, _t, _i, **kwargs):
    self.spike[:] = False
    self._run(_t)


class SpikeTimeInput(SpikeTimeGroup):
  pass


class PoissonGroup(NeuGroup):
  """Poisson Neuron Group.
  """

  def __init__(self, size, freqs, seed=None, name=None):
    super(PoissonGroup, self).__init__(size=size, name=name)

    self.freqs = freqs
    self.dt = bm.get_dt() / 1000.
    self.size = (size,) if isinstance(size, int) else tuple(size)
    self.spike = bm.Variable(jnp.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(jnp.ones(self.num) * -1e7)
    self.rng = bm.random.RandomState(seed=seed)

  def update(self, _t, _i):
    self.spike.update(self.rng.random(self.num) <= self.freqs * self.dt)
    self.t_last_spike.update(bm.where(self.spike, _t, self.t_last_spike))


class PoissonInput(PoissonGroup):
  pass
