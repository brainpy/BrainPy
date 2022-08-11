# -*- coding: utf-8 -*-


from jax import custom_gradient, custom_jvp

from brainpy.math import numpy_ops as bm
from brainpy.math.jaxarray import JaxArray
from brainpy.types import Array

from brainpy.math.setting import dftype

__all__ = [
  'spike_with_sigmoid_grad',
  'spike_with_linear_grad',
  'spike_with_gaussian_grad',
  'spike_with_mg_grad',

  'spike2_with_sigmoid_grad',
  'spike2_with_linear_grad',
  'step_pwl'
]


def _consistent_type(target, compare):
  return target.value if not isinstance(compare, JaxArray) else target


@custom_gradient
def spike_with_sigmoid_grad(x: Array, scale: float = None):
  """Spike function with the sigmoid surrogate gradient.

  Parameters
  ----------
  x: Array
    The input data.
  scale: float
    The scaling factor.
  """
  z = bm.asarray(x >= 0, dtype=dftype())

  def grad(dE_dz):
    _scale = 100. if scale is None else scale
    dE_dx = dE_dz / (_scale * bm.abs(x) + 1.0) ** 2
    if scale is None:
      return (_consistent_type(dE_dx, x),)
    else:
      dscale = bm.zeros_like(_scale)
      return (_consistent_type(dE_dx, x),
              _consistent_type(dscale, scale))

  return z, grad


@custom_gradient
def spike2_with_sigmoid_grad(x_new: Array, x_old: Array, scale: float = None):
  """Spike function with the sigmoid surrogate gradient.

  Parameters
  ----------
  x_new: Array
    The input data.
  x_old: Array
    The input data.
  scale: optional, float
    The scaling factor.
  """
  x_new_comp = x_new >= 0
  x_old_comp = x_old < 0
  z = bm.asarray(bm.logical_and(x_new_comp, x_old_comp), dtype=dftype())

  def grad(dE_dz):
    _scale = 100. if scale is None else scale
    dx_new = (dE_dz / (_scale * bm.abs(x_new) + 1.0) ** 2) * bm.asarray(x_old_comp, dtype=dftype())
    dx_old = -(dE_dz / (_scale * bm.abs(x_old) + 1.0) ** 2) * bm.asarray(x_new_comp, dtype=dftype())
    if scale is None:
      return (_consistent_type(dx_new, x_new),
              _consistent_type(dx_old, x_old))
    else:
      dscale = bm.zeros_like(_scale)
      return (_consistent_type(dx_new, x_new),
              _consistent_type(dx_old, x_old),
              _consistent_type(dscale, scale))

  return z, grad


@custom_gradient
def spike_with_linear_grad(x: Array, scale: float = None):
  """Spike function with the relu surrogate gradient.

  Parameters
  ----------
  x: Array
    The input data.
  scale: float
    The scaling factor.
  """
  z = bm.asarray(x >= 0., dtype=dftype())

  def grad(dE_dz):
    _scale = 0.3 if scale is None else scale
    dE_dx = dE_dz * bm.maximum(1 - bm.abs(x), 0) * _scale
    if scale is None:
      return (_consistent_type(dE_dx, x),)
    else:
      dscale = bm.zeros_like(_scale)
      return (_consistent_type(dE_dx, x), _consistent_type(dscale, _scale))

  return z, grad


@custom_gradient
def spike2_with_linear_grad(x_new: Array, x_old: Array, scale: float = 10.):
  """Spike function with the linear surrogate gradient.

  Parameters
  ----------
  x_new: Array
    The input data.
  x_old: Array
    The input data.
  scale: float
    The scaling factor.
  """
  x_new_comp = x_new >= 0
  x_old_comp = x_old < 0
  z = bm.asarray(bm.logical_and(x_new_comp, x_old_comp), dtype=dftype())

  def grad(dE_dz):
    _scale = 0.3 if scale is None else scale
    dx_new = (dE_dz * bm.maximum(1 - bm.abs(x_new), 0) * _scale) * bm.asarray(x_old_comp, dtype=dftype())
    dx_old = -(dE_dz * bm.maximum(1 - bm.abs(x_old), 0) * _scale) * bm.asarray(x_new_comp, dtype=dftype())
    if scale is None:
      return (_consistent_type(dx_new, x_new),
              _consistent_type(dx_old, x_old))
    else:
      dscale = bm.zeros_like(_scale)
      return (_consistent_type(dx_new, x_new),
              _consistent_type(dx_old, x_old),
              _consistent_type(dscale, scale))

  return z, grad


def _gaussian(x, mu, sigma):
  return bm.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / bm.sqrt(2 * bm.pi) / sigma


@custom_gradient
def spike_with_gaussian_grad(x, sigma=None, scale=None):
  """Spike function with the Gaussian surrogate gradient.
  """
  z = bm.asarray(x >= 0., dtype=dftype())

  def grad(dE_dz):
    _scale = 0.5 if scale is None else scale
    _sigma = 0.5 if sigma is None else sigma
    dE_dx = dE_dz * _gaussian(x, 0., _sigma) * _scale
    returns = (_consistent_type(dE_dx, x),)
    if sigma is not None:
      returns += (_consistent_type(bm.zeros_like(_sigma), sigma), )
    if scale is not None:
      returns += (_consistent_type(bm.zeros_like(_scale), scale), )
    return returns

  return z, grad


@custom_gradient
def spike_with_mg_grad(x, h=None, s=None, sigma=None, scale=None):
  """Spike function with the multi-Gaussian surrogate gradient.

  Parameters
  ----------
  x: ndarray
    The variable to judge spike.
  h: float
    The hyper-parameters of approximate function
  s: float
    The hyper-parameters of approximate function
  sigma: float
    The gaussian sigma.
  scale: float
    The gradient scale.
  """
  z = bm.asarray(x >= 0., dtype=dftype())

  def grad(dE_dz):
    _sigma = 0.5 if sigma is None else sigma
    _scale = 0.5 if scale is None else scale
    _s = 6.0 if s is None else s
    _h = 0.15 if h is None else h
    dE_dx = dE_dz * (_gaussian(x, mu=0., sigma=_sigma) * (1. + _h)
                     - _gaussian(x, mu=_sigma, sigma=_s * _sigma) * _h
                     - _gaussian(x, mu=-_sigma, sigma=_s * _sigma) * _h) * _scale
    returns = (_consistent_type(dE_dx, x),)
    if h is not None:
      returns += (_consistent_type(bm.zeros_like(_h), h),)
    if s is not None:
      returns += (_consistent_type(bm.zeros_like(_s), s),)
    if sigma is not None:
      returns += (_consistent_type(bm.zeros_like(_sigma), sigma),)
    if scale is not None:
      returns += (_consistent_type(bm.zeros_like(_scale), scale),)
    return returns

  return z, grad


@custom_jvp
def step_pwl(x, threshold, window=0.5, max_spikes_per_dt: int = bm.inf):
  """
  Heaviside step function with piece-wise linear derivative to use as spike-generation surrogate

  Args:
      x (float):          Input value
      threshold (float):  Firing threshold
      window (float): Learning window around threshold. Default: 0.5
      max_spikes_per_dt (int): Maximum number of spikes that may be produced each dt. Default: ``np.inf``, do not clamp spikes

  Returns:
      float: Number of output events for each input value
  """
  spikes = (x >= threshold) * bm.floor(x / threshold)
  return bm.clip(spikes, 0.0, max_spikes_per_dt)


@step_pwl.defjvp
def step_pwl_jvp(primals, tangents):
  x, threshold, window, max_spikes_per_dt = primals
  x_dot, threshold_dot, window_dot, max_spikes_per_dt_dot = tangents
  primal_out = step_pwl(*primals)
  tangent_out = (x >= (threshold - window)) * (x_dot / threshold - threshold_dot * x / (threshold ** 2))
  return primal_out, tangent_out
