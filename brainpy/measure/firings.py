# -*- coding: utf-8 -*-

import numpy as onp
import jax.numpy as jnp

from brainpy import math as bm

__all__ = [
  'raster_plot',
  'firing_rate',
]


def raster_plot(sp_matrix, times):
  """Get spike raster plot which displays the spiking activity
  of a group of neurons over time.

  Parameters
  ----------
  sp_matrix : bnp.ndarray
      The matrix which record spiking activities.
  times : bnp.ndarray
      The time steps.

  Returns
  -------
  raster_plot : tuple
      Include (neuron index, spike time).
  """
  sp_matrix = bm.as_numpy(sp_matrix)
  times = onp.asarray(times)
  elements = onp.where(sp_matrix > 0.)
  index = elements[1]
  time = times[elements[0]]
  return index, time


def firing_rate(spikes, width, dt=None, numpy=True):
  r"""Calculate the mean firing rate over in a neuron group.

  This method is adopted from Brian2.

  The firing rate in trial :math:`k` is the spike count :math:`n_{k}^{sp}`
  in an interval of duration :math:`T` divided by :math:`T`:

  .. math::

      v_k = {n_k^{sp} \over T}

  Parameters
  ----------
  spikes : ndarray
    The spike matrix which record spiking activities.
  width : int, float
    The width of the ``window`` in millisecond.
  dt : float, optional
    The sample rate.
  numpy: bool
    Whether we use numpy array as the functional output.
    If ``False``, this function can be JIT compiled.

  Returns
  -------
  rate : ndarray
      The population rate in Hz, smoothed with the given window.
  """
  spikes = bm.as_numpy(spikes) if numpy else bm.as_device_array(spikes)
  np = onp if numpy else jnp
  dt = bm.get_dt() if (dt is None) else dt
  width1 = int(width / 2 / dt) * 2 + 1
  window = np.ones(width1) * 1000 / width
  return np.convolve(np.mean(spikes, axis=1), window, mode='same')
