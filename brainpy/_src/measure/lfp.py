# -*- coding: utf-8 -*-


from jax import numpy as jnp

import brainpy._src.math as bm

__all__ = [
  'unitary_LFP',
]


def unitary_LFP(times, spikes, spike_type='exc',
                xmax=0.2, ymax=0.2, va=200., lambda_=0.2,
                sig_i=2.1, sig_e=2.1 * 1.5, location='soma layer', seed=None):
  """A kernel-based method to calculate unitary local field potentials (uLFP)
  from a network of spiking neurons [1]_.

  .. note::
     This method calculates LFP only from the neuronal spikes. It does not consider
     the subthreshold synaptic events, or the dendritic voltage-dependent ion channels.

  Examples
  --------

  If you have spike data of excitatory and inhibtiory neurons, you can get the LFP
  by the following methods:

  >>> import brainpy as bp
  >>> n_time = 1000
  >>> n_exc = 100
  >>> n_inh = 25
  >>> times = bm.arange(n_time) * 0.1
  >>> exc_sps = bp.math.random.random((n_time, n_exc)) < 0.3
  >>> inh_sps = bp.math.random.random((n_time, n_inh)) < 0.4
  >>> lfp = bp.measure.unitary_LFP(times, exc_sps, 'exc')
  >>> lfp += bp.measure.unitary_LFP(times, inh_sps, 'inh')

  Parameters
  ----------
  times: ndarray
    The times of the recording points.
  spikes: ndarray
    The spikes of excitatory neurons recorded by brainpy monitors.
  spike_type: str
    The neuron type of the spike trains. It can be "exc" or "inh".
  location: str
    The location of the spikes recorded. It can be "soma layer", "deep layer",
    "superficial layer" and "surface".
  xmax: float
    Size of the array (in mm).
  ymax: float
    Size of the array (in mm).
  va: int, float
    The axon velocity (mm/sec).
  lambda_: float
    The space constant (mm).
  sig_i: float
    The std-dev of inhibition (in ms)
  sig_e: float
    The std-dev for excitation (in ms).
  seed: int
    The random seed.

  References
  ----------
  .. [1] Telenczuk, Bartosz, Maria Telenczuk, and Alain Destexhe. "A kernel-based
         method to calculate local field potentials from networks of spiking
         neurons." Journal of Neuroscience Methods 344 (2020): 108871.

  """
  times = bm.as_jax(times)
  spikes = bm.as_jax(spikes)
  if spike_type not in ['exc', 'inh']:
    raise ValueError('"spike_type" should be "exc or ""inh". ')
  if spikes.ndim != 2:
    raise ValueError('"E_spikes" should be a matrix with shape of (num_time, num_neuron). '
                     f'But we got {spikes.shape}')
  if times.shape[0] != spikes.shape[0]:
    raise ValueError('times and spikes should be consistent at the firs axis. '
                     f'Bug we got {times.shape[0]} != {spikes.shape}.')

  # Distributing cells in a 2D grid
  rng = bm.random.RandomState(seed)
  num_neuron = spikes.shape[1]
  pos_xs, pos_ys = rng.rand(2, num_neuron).value * jnp.array([[xmax], [ymax]])
  pos_xs, pos_ys = jnp.asarray(pos_xs), jnp.asarray(pos_ys)

  # distance/coordinates
  xe, ye = xmax / 2, ymax / 2  # coordinates of electrode
  dist = jnp.sqrt((pos_xs - xe) ** 2 + (pos_ys - ye) ** 2)  # distance to electrode in mm

  # amplitude
  if location == 'soma layer':
    amp_e, amp_i = 0.48, 3.  # exc/inh uLFP amplitude (soma layer)
  elif location == 'deep layer':
    amp_e, amp_i = -0.16, -0.2  # exc/inh uLFP amplitude (deep layer)
  elif location == 'superficial layer':
    amp_e, amp_i = 0.24, -1.2  # exc/inh uLFP amplitude (superficial layer)
  elif location == 'surface layer':
    amp_e, amp_i = -0.08, 0.3  # exc/inh uLFP amplitude (surface)
  else:
    raise NotImplementedError
  A = jnp.exp(-dist / lambda_) * (amp_e if spike_type == 'exc' else amp_i)

  # delay
  delay = 10.4 + dist / va  # delay to peak (in ms)

  # LFP Calculation
  iis, ids = jnp.where(spikes)
  tts = times[iis] + delay[ids]
  exc_amp = A[ids]
  tau = (2 * sig_e * sig_e) if spike_type == 'exc' else (2 * sig_i * sig_i)
  return bm.for_loop(lambda t: jnp.sum(exc_amp * jnp.exp(-(t - tts) ** 2 / tau)), times)
