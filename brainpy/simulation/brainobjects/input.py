# -*- coding: utf-8 -*-

import numpy as np

from brainpy import errors, math
from brainpy.simulation.brainobjects.neuron import NeuGroup


__all__ = [
  'SpikeTimeInput',
  'PoissonInput',
]


class SpikeTimeInput(NeuGroup):
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
  indices : int, list, tuple
      The neuron indices at each time point to emit spikes.
  times : list, np.ndarray
      The time points which generate the spikes.
  steps : tuple of str, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  monitors : None, list, tuple, datastructures.Monitor
      Variables to monitor.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, size, times, indices, need_sort=True, **kwargs):
    super(SpikeTimeInput, self).__init__(size=size, **kwargs)

    if len(indices) != len(times):
      raise errors.BrainPyError(f'The length of "indices" and "times" must be the same. '
                                f'However, we got {len(indices)} != {len(times)}.')

    # data about times and indices
    self.idx = 0
    self.times = math.asarray(times, dtype=math.float_)
    self.indices = np.asarray(indices, dtype=math.int_)
    self.num_times = len(times)
    if need_sort:
      sort_idx = np.argsort(times)
      self.indices = self.indices[sort_idx]
    self.spike = math.zeros(self.num, dtype=bool)

  def update(self, _t, _i, **kwargs):
    self.spike[:] = False
    while self.idx < self.num_times and _t >= self.times[self.idx]:
      self.spike[self.indices[self.idx]] = 1.
      self.idx += 1


class PoissonInput(NeuGroup):
  """Poisson Neuron Group.

  Parameters
  ----------
  steps : tuple of str, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  monitors : None, list, tuple, datastructures.Monitor
      Variables to monitor.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, size, freqs, seed=None, **kwargs):
    super(PoissonInput, self).__init__(size=size, **kwargs)

    self.freqs = freqs
    self.dt = math.get_dt() / 1000.
    self.size = (size,) if isinstance(size, int) else tuple(size)
    self.spike = math.Variable(math.zeros(self.num, dtype=bool))
    self.t_last_spike = math.Variable(math.ones(self.num) * -1e7)
    self.rng = math.random.RandomState(seed=seed)

  def update(self, _t, _i, **kwargs):
    self.spike[:] = self.rng.random(self.num) <= self.freqs * self.dt
    self.t_last_spike[:] = math.where(self.spike, _t, self.t_last_spike)

