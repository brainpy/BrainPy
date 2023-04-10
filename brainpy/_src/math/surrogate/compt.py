# -*- coding: utf-8 -*-

import warnings

from jax import custom_gradient, numpy as jnp

from brainpy._src.math.compat_numpy import asarray
from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.environment import get_float
from brainpy._src.math.ndarray import Array

__all__ = [
  'spike_with_sigmoid_grad',
  'spike_with_linear_grad',
  'spike_with_gaussian_grad',
  'spike_with_mg_grad',

  'spike2_with_sigmoid_grad',
  'spike2_with_linear_grad',
]


def _consistent_type(target, compare):
  return as_jax(target) if not isinstance(compare, Array) else asarray(target)


@custom_gradient
def spike_with_sigmoid_grad(x: Array, scale: float = 100.):
  """Spike function with the sigmoid surrogate gradient.

  .. deprecated:: 2.3.1
     Please use ``brainpy.math.surrogate.sigmoid_grad()`` instead.
     Will be removed after version 2.4.0.

  Parameters
  ----------
  x: Array
    The input data.
  scale: float
    The scaling factor.
  """
  warnings.warn('Use `brainpy.math.surrogate.inv_square_grad()` instead.', UserWarning)

  x = as_jax(x)
  z = jnp.asarray(x >= 0, dtype=get_float())

  def grad(dE_dz):
    dE_dz = as_jax(dE_dz)
    dE_dx = dE_dz / (scale * jnp.abs(x) + 1.0) ** 2
    if scale is None:
      return (_consistent_type(dE_dx, x),)
    else:
      dscale = jnp.zeros_like(scale)
      return (dE_dx, dscale)

  return z, grad


@custom_gradient
def spike2_with_sigmoid_grad(x_new: Array, x_old: Array, scale: float = None):
  """Spike function with the sigmoid surrogate gradient.

  .. deprecated:: 2.3.1
     Please use ``brainpy.math.surrogate.inv_square_grad2()`` instead.
     Will be removed after version 2.4.0.

  Parameters
  ----------
  x_new: Array
    The input data.
  x_old: Array
    The input data.
  scale: optional, float
    The scaling factor.
  """
  warnings.warn('Use `brainpy.math.surrogate.inv_square_grad2()` instead.', UserWarning)

  x_new_comp = x_new >= 0
  x_old_comp = x_old < 0
  z = jnp.asarray(jnp.logical_and(x_new_comp, x_old_comp), dtype=get_float())

  def grad(dE_dz):
    _scale = 100. if scale is None else scale
    dx_new = (dE_dz / (_scale * jnp.abs(x_new) + 1.0) ** 2) * jnp.asarray(x_old_comp, dtype=get_float())
    dx_old = -(dE_dz / (_scale * jnp.abs(x_old) + 1.0) ** 2) * jnp.asarray(x_new_comp, dtype=get_float())
    if scale is None:
      return (_consistent_type(dx_new, x_new),
              _consistent_type(dx_old, x_old))
    else:
      dscale = jnp.zeros_like(_scale)
      return (_consistent_type(dx_new, x_new),
              _consistent_type(dx_old, x_old),
              _consistent_type(dscale, scale))

  return z, grad


@custom_gradient
def spike_with_linear_grad(x: Array, scale: float = None):
  """Spike function with the relu surrogate gradient.

  .. deprecated:: 2.3.1
     Please use ``brainpy.math.surrogate.relu_grad()`` instead.
     Will be removed after version 2.4.0.

  Parameters
  ----------
  x: Array
    The input data.
  scale: float
    The scaling factor.
  """

  warnings.warn('Use `brainpy.math.surrogate.relu_grad()` instead.', UserWarning)

  z = jnp.asarray(x >= 0., dtype=get_float())

  def grad(dE_dz):
    _scale = 0.3 if scale is None else scale
    dE_dx = dE_dz * jnp.maximum(1 - jnp.abs(x), 0) * _scale
    if scale is None:
      return (_consistent_type(dE_dx, x),)
    else:
      dscale = jnp.zeros_like(_scale)
      return (_consistent_type(dE_dx, x), _consistent_type(dscale, _scale))

  return z, grad


@custom_gradient
def spike2_with_linear_grad(x_new: Array, x_old: Array, scale: float = 10.):
  """Spike function with the linear surrogate gradient.

  .. deprecated:: 2.3.1
     Please use ``brainpy.math.surrogate.relu_grad2()`` instead.
     Will be removed after version 2.4.0.

  Parameters
  ----------
  x_new: Array
    The input data.
  x_old: Array
    The input data.
  scale: float
    The scaling factor.
  """
  warnings.warn('Use `brainpy.math.surrogate.relu_grad2()` instead.', UserWarning)

  x_new_comp = x_new >= 0
  x_old_comp = x_old < 0
  z = jnp.asarray(jnp.logical_and(x_new_comp, x_old_comp), dtype=get_float())

  def grad(dE_dz):
    _scale = 0.3 if scale is None else scale
    dx_new = (dE_dz * jnp.maximum(1 - jnp.abs(x_new), 0) * _scale) * jnp.asarray(x_old_comp, dtype=get_float())
    dx_old = -(dE_dz * jnp.maximum(1 - jnp.abs(x_old), 0) * _scale) * jnp.asarray(x_new_comp, dtype=get_float())
    if scale is None:
      return (_consistent_type(dx_new, x_new),
              _consistent_type(dx_old, x_old))
    else:
      dscale = jnp.zeros_like(_scale)
      return (_consistent_type(dx_new, x_new),
              _consistent_type(dx_old, x_old),
              _consistent_type(dscale, scale))

  return z, grad


def _gaussian(x, mu, sigma):
  return jnp.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / jnp.sqrt(2 * jnp.pi) / sigma


@custom_gradient
def spike_with_gaussian_grad(x, sigma=None, scale=None):
  """Spike function with the Gaussian surrogate gradient.

  .. deprecated:: 2.3.1
     Please use ``brainpy.math.surrogate.gaussian_grad()`` instead.
     Will be removed after version 2.4.0.

  """

  warnings.warn('Use `brainpy.math.surrogate.gaussian_grad()` instead.', UserWarning)

  z = jnp.asarray(x >= 0., dtype=get_float())

  def grad(dE_dz):
    _scale = 0.5 if scale is None else scale
    _sigma = 0.5 if sigma is None else sigma
    dE_dx = dE_dz * _gaussian(x, 0., _sigma) * _scale
    returns = (_consistent_type(dE_dx, x),)
    if sigma is not None:
      returns += (_consistent_type(jnp.zeros_like(_sigma), sigma),)
    if scale is not None:
      returns += (_consistent_type(jnp.zeros_like(_scale), scale),)
    return returns

  return z, grad


@custom_gradient
def spike_with_mg_grad(x, h=None, s=None, sigma=None, scale=None):
  """Spike function with the multi-Gaussian surrogate gradient.

  .. deprecated:: 2.3.1
     Please use ``brainpy.math.surrogate.multi_sigmoid_grad()`` instead.
     Will be removed after version 2.4.0.

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

  warnings.warn('Use `brainpy.math.surrogate.multi_sigmoid_grad()` instead.', UserWarning)

  z = jnp.asarray(x >= 0., dtype=get_float())

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
      returns += (_consistent_type(jnp.zeros_like(_h), h),)
    if s is not None:
      returns += (_consistent_type(jnp.zeros_like(_s), s),)
    if sigma is not None:
      returns += (_consistent_type(jnp.zeros_like(_sigma), sigma),)
    if scale is not None:
      returns += (_consistent_type(jnp.zeros_like(_scale), scale),)
    return returns

  return z, grad

