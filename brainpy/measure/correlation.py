# -*- coding: utf-8 -*-

from functools import partial

import numpy as np
from jax import vmap, jit, lax, numpy as jnp

from brainpy import math as bm

__all__ = [
  'cross_correlation',
  'voltage_fluctuation',
  'matrix_correlation',
  'weighted_correlation',
  'functional_connectivity',
  # 'functional_connectivity_dynamics',
]


# @jit
@partial(vmap, in_axes=(None, 0, 0))
def _cc(states, i, j):
  sqrt_ij = jnp.sqrt(jnp.sum(states[i]) * jnp.sum(states[j]))
  return lax.cond(sqrt_ij == 0.,
                  lambda _: 0.,
                  lambda _: jnp.sum(states[i] * states[j]) / sqrt_ij,
                  None)


def cross_correlation(spikes, bin, dt=None):
  r"""Calculate cross correlation index between neurons.

  The coherence [1]_ between two neurons i and j is measured by their
  cross-correlation of spike trains at zero time lag within a time bin
  of :math:`\Delta t = \tau`. More specifically, suppose that a long
  time interval T is divided into small bins of :math:`\Delta t` and
  that two spike trains are given by :math:`X(l)=` 0 or 1, :math:`Y(l)=` 0
  or 1, :math:`l=1,2, \ldots, K(T / K=\tau)`. Thus, we define a coherence
  measure for the pair as:

  .. math::

      \kappa_{i j}(\tau)=\frac{\sum_{l=1}^{K} X(l) Y(l)}
      {\sqrt{\sum_{l=1}^{K} X(l) \sum_{l=1}^{K} Y(l)}}

  The population coherence measure :math:`\kappa(\tau)` is defined by the
  average of :math:`\kappa_{i j}(\tau)` over many pairs of neurons in the
  network.

  Parameters
  ----------
  spikes :
      The history of spike states of the neuron group.
  bin : float, int
      The time bin to normalize spike states.
  dt : float, optional
      The time precision.

  Returns
  -------
  cc_index : float
      The cross correlation value which represents the synchronization index.

  References
  ----------
  .. [1] Wang, Xiao-Jing, and György Buzsáki. "Gamma oscillation by synaptic
         inhibition in a hippocampal interneuronal network model." Journal of
         neuroscience 16.20 (1996): 6402-6413.
  """
  spikes = bm.as_device_array(spikes)
  dt = bm.get_dt() if dt is None else dt
  bin_size = int(bin / dt)
  num_hist, num_neu = spikes.shape
  num_bin = int(np.ceil(num_hist / bin_size))
  if num_bin * bin_size != num_hist:
    spikes = jnp.append(spikes, jnp.zeros((num_bin * bin_size - num_hist, num_neu)), axis=0)
  states = spikes.T.reshape((num_neu, num_bin, bin_size))
  states = jnp.asarray(jnp.sum(states, axis=2) > 0., dtype=jnp.float_)
  indices = jnp.tril_indices(num_neu, k=-1)
  return jnp.mean(_cc(states, *indices))


@partial(vmap, in_axes=(None, 0))
def _var(neu_signal, i):
  neu_signal = neu_signal[:, i]
  return jnp.mean(neu_signal * neu_signal) - jnp.mean(neu_signal) ** 2


# @jit
def voltage_fluctuation(potentials):
  r"""Calculate neuronal synchronization via voltage variance.

  The method comes from [1]_ [2]_ [3]_.

  First, average over the membrane potential :math:`V`

  .. math::

      V(t) = \frac{1}{N} \sum_{i=1}^{N} V_i(t)

  The variance of the time fluctuations of :math:`V(t)` is

  .. math::

      \sigma_V^2 = \left\langle \left[ V(t) \right]^2 \right\rangle_t -
      \left[ \left\langle V(t) \right\rangle_t \right]^2

  where :math:`\left\langle \ldots \right\rangle_t = (1 / T_m) \int_0^{T_m} dt \, \ldots`
  denotes time-averaging over a large time, :math:`\tau_m`. After normalization
  of :math:`\sigma_V` to the average over the population of the single cell
  membrane potentials

  .. math::

      \sigma_{V_i}^2 = \left\langle\left[ V_i(t) \right]^2 \right\rangle_t -
      \left[ \left\langle V_i(t) \right\rangle_t \right]^2

  one defines a synchrony measure, :math:`\chi (N)`, for the activity of a system
  of :math:`N` neurons by:

  .. math::

      \chi^2 \left( N \right) = \frac{\sigma_V^2}{ \frac{1}{N} \sum_{i=1}^N
      \sigma_{V_i}^2}

  Parameters
  ----------
  potentials :
      The membrane potential matrix of the neuron group.

  Returns
  -------
  sync_index : float
      The synchronization index.

  References
  ----------
  .. [1] Golomb, D. and Rinzel J. (1993) Dynamics of globally coupled
         inhibitory neurons with heterogeneity. Phys. Rev. reversal_potential 48:4810-4814.
  .. [2] Golomb D. and Rinzel J. (1994) Clustering in globally coupled
         inhibitory neurons. Physica D 72:259-282.
  .. [3] David Golomb (2007) Neuronal synchrony measures. Scholarpedia, 2(1):1347.
  """

  potentials = bm.as_device_array(potentials)
  num_hist, num_neu = potentials.shape
  var_mean = jnp.mean(_var(potentials, jnp.arange(num_neu)))
  avg = jnp.mean(potentials, axis=1)
  avg_var = jnp.mean(avg * avg) - jnp.mean(avg) ** 2
  return lax.cond(var_mean != 0., lambda _: avg_var / var_mean, lambda _: 1., None)


def matrix_correlation(x, y):
  """Pearson correlation of the lower triagonal of two matrices.

    The triangular matrix is offset by k = 1 in order to ignore the diagonal line

  Parameters
  ----------
  x: tensor
    First matrix.
  y: tensor
    Second matrix

  Returns
  -------
  coef: tensor
    Correlation coefficient
  """
  x = bm.as_numpy(x)
  y = bm.as_numpy(y)
  if x.ndim != 2:
    raise ValueError(f'Only support 2d tensor, but we got a tensor '
                     f'with the shape of {x.shape}')
  if y.ndim != 2:
    raise ValueError(f'Only support 2d tensor, but we got a tensor '
                     f'with the shape of {y.shape}')
  x = x[np.triu_indices_from(x, k=1)]
  y = y[np.triu_indices_from(y, k=1)]
  cc = np.corrcoef(x, y)[0, 1]
  return cc


def functional_connectivity(activities):
  """Functional connectivity matrix of timeseries activities.

  Parameters
  ----------
  activities: tensor
    The multidimensional tensor with the shape of ``(num_time, num_sample)``.

  Returns
  -------
  connectivity_matrix: tensor
    ``num_sample x num_sample`` functional connectivity matrix.
  """
  activities = bm.as_numpy(activities)
  if activities.ndim != 2:
    raise ValueError('Only support 2d tensor with shape of "(num_time, num_sample)". '
                     f'But we got a tensor with the shape of {activities.shape}')
  fc = np.corrcoef(activities.T)
  return np.nan_to_num(fc)


# @jit
def functional_connectivity_dynamics(activities, window_size=30, step_size=5):
  """Computes functional connectivity dynamics (FCD) matrix.

  Parameters
  ----------
  activities: tensor
    The time series with shape of ``(num_time, num_sample)``.
  window_size: int
    Size of each rolling window in time steps, defaults to 30.
  step_size: int
    Step size between each rolling window, defaults to 5.

  Returns
  -------
  fcd_matrix: tensor
    FCD matrix.
  """
  pass


def _weighted_mean(x, w):
  """Weighted Mean"""
  return jnp.sum(x * w) / jnp.sum(w)


def _weighted_cov(x, y, w):
  """Weighted Covariance"""
  return jnp.sum(w * (x - _weighted_mean(x, w)) * (y - _weighted_mean(y, w))) / jnp.sum(w)


# @jit
def weighted_correlation(x, y, w):
  """Weighted Pearson correlation of two data series.

  Parameters
  ----------
  x: tensor
    The data series 1.
  y: tensor
    The data series 2.
  w: tensor
    Weight vector, must have same length as x and y.

  Returns
  -------
  corr: tensor
    Weighted correlation coefficient.
  """
  x = bm.as_device_array(x)
  y = bm.as_device_array(y)
  w = bm.as_device_array(w)
  if x.ndim != 1:
    raise ValueError(f'Only support 1d tensor, but we got a tensor '
                     f'with the shape of {x.shape}')
  if y.ndim != 1:
    raise ValueError(f'Only support 1d tensor, but we got a tensor '
                     f'with the shape of {y.shape}')
  if w.ndim != 1:
    raise ValueError(f'Only support 1d tensor, but we got a tensor '
                     f'with the shape of {w.shape}')
  return _weighted_cov(x, y, w) / jnp.sqrt(_weighted_cov(x, x, w) * _weighted_cov(y, y, w))
