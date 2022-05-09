# -*- coding: utf-8 -*-

import numpy as np
import jax.numpy as jnp
from jax import jit

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
  times = np.asarray(times)
  elements = np.where(sp_matrix > 0.)
  index = elements[1]
  time = times[elements[0]]
  return index, time


@jit
def _firing_rate(sp_matrix, window):
  sp_matrix = bm.as_device_array(sp_matrix)
  rate = jnp.sum(sp_matrix, axis=1) / sp_matrix.shape[1]
  return jnp.convolve(rate, window, mode='same')


def firing_rate(sp_matrix, width, dt=None, numpy=True):
  r"""Calculate the mean firing rate over in a neuron group.

  This method is adopted from Brian2.

  The firing rate in trial :math:`k` is the spike count :math:`n_{k}^{sp}`
  in an interval of duration :math:`T` divided by :math:`T`:

  .. math::

      v_k = {n_k^{sp} \over T}

  Parameters
  ----------
  sp_matrix : math.JaxArray, np.ndarray
    The spike matrix which record spiking activities.
  width : int, float
    The width of the ``window`` in millisecond.
  dt : float, optional
    The sample rate.

  Returns
  -------
  rate : numpy.ndarray
      The population rate in Hz, smoothed with the given window.
  """
  dt = bm.get_dt() if (dt is None) else dt
  width1 = int(width / 2 / dt) * 2 + 1
  window = jnp.ones(width1) * 1000 / width
  fr = _firing_rate(sp_matrix, window)
  return bm.as_numpy(fr) if numpy else fr

