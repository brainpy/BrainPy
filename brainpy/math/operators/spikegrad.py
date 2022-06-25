# -*- coding: utf-8 -*-


from jax import custom_gradient, custom_jvp

from brainpy.math import numpy_ops as bm
from brainpy.math.jaxarray import JaxArray
from brainpy.types import Tensor

from brainpy.math.setting import dftype

__all__ = [
  'spike_with_sigmoid_grad',
  'spike2_with_sigmoid_grad',
  'spike_with_relu_grad',
  'spike2_with_relu_grad',
  'step_pwl'
]


def _consistent_type(target, compare):
  return target.value if not isinstance(compare, JaxArray) else target


@custom_gradient
def spike_with_sigmoid_grad(x: Tensor, scale: float = None):
  """Spike function with the sigmoid surrogate gradient.

  Parameters
  ----------
  x: Tensor
    The input data.
  scale: float
    The scaling factor.
  """
  z = bm.asarray(x >= 0, dtype=dftype())

  def grad(dE_dz):
    _scale = scale
    if scale is None:
      _scale = 100.
    dE_dx = dE_dz / (_scale * bm.abs(x) + 1.0) ** 2
    if scale is None:
      return (_consistent_type(dE_dx, x),)
    else:
      dscale = bm.zeros_like(_scale)
      return (_consistent_type(dE_dx, x),
              _consistent_type(dscale, scale))

  return z, grad


@custom_gradient
def spike2_with_sigmoid_grad(x_new: Tensor, x_old: Tensor, scale: float = None):
  """Spike function with the sigmoid surrogate gradient.

  Parameters
  ----------
  x_new: Tensor
    The input data.
  x_old: Tensor
    The input data.
  scale: optional, float
    The scaling factor.
  """
  x_new_comp = x_new >= 0
  x_old_comp = x_old < 0
  z = bm.asarray(bm.logical_and(x_new_comp, x_old_comp), dtype=dftype())

  def grad(dE_dz):
    _scale = scale
    if scale is None:
      _scale = 100.
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
def spike_with_relu_grad(x: Tensor, scale: float = None):
  """Spike function with the relu surrogate gradient.

  Parameters
  ----------
  x: Tensor
    The input data.
  scale: float
    The scaling factor.
  """
  z = bm.asarray(x >= 0., dtype=dftype())

  def grad(dE_dz):
    _scale = scale
    if scale is None:  _scale = 0.3
    dE_dx = dE_dz * bm.maximum(1 - bm.abs(x), 0) * _scale
    if scale is None:
      return (_consistent_type(dE_dx, x),)
    else:
      dscale = bm.zeros_like(_scale)
      return (_consistent_type(dE_dx, x),
              _consistent_type(dscale, _scale))

  return z, grad


@custom_gradient
def spike2_with_relu_grad(x_new: Tensor, x_old: Tensor, scale: float = 10.):
  """Spike function with the relu surrogate gradient.

  Parameters
  ----------
  x_new: Tensor
    The input data.
  x_old: Tensor
    The input data.
  scale: float
    The scaling factor.
  """
  x_new_comp = x_new >= 0
  x_old_comp = x_old < 0
  z = bm.asarray(bm.logical_and(x_new_comp, x_old_comp), dtype=dftype())

  def grad(dE_dz):
    _scale = scale
    if scale is None:
      _scale = 0.3
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
