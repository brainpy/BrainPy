# -*- coding: utf-8 -*-


from typing import Union

import jax
import jax.numpy as jnp

import brainpy.math.numpy_ops as bm
from brainpy.math.ndarray import Array
from ._utils import generalized_vjp_custom

__all__ = [
  'inv_square_grad2',
  'relu_grad2',
]


@generalized_vjp_custom('x_new', 'x_old', alpha=100.)
def inv_square_grad2(
    x_new: Union[jax.Array, Array],
    x_old: Union[jax.Array, Array],
    alpha: float
):
  x_new_comp = x_new >= 0
  x_old_comp = x_old < 0
  z = jnp.asarray(jnp.logical_and(x_new_comp, x_old_comp), dtype=x_new.dtype)

  def grad(dz):
    dz = bm.as_jax(dz)
    dx_new = (dz / (alpha * jnp.abs(x_new) + 1.0) ** 2) * jnp.asarray(x_old_comp, dtype=x_old.dtype)
    dx_old = -(dz / (alpha * jnp.abs(x_old) + 1.0) ** 2) * jnp.asarray(x_new_comp, dtype=x_new.dtype)
    return dx_new, dx_old

  return z, grad


@generalized_vjp_custom('x_new', 'x_old', alpha=.3, width=1.)
def relu_grad2(
    x_new: Union[jax.Array, Array],
    x_old: Union[jax.Array, Array],
    alpha: float,
    width: float,
):
  x_new_comp = x_new >= 0
  x_old_comp = x_old < 0
  z = jnp.asarray(jnp.logical_and(x_new_comp, x_old_comp), dtype=x_new.dtype)

  def grad(dz):
    dz = bm.as_jax(dz)
    dx_new = (dz * jnp.maximum(width - jnp.abs(x_new), 0) * alpha) * jnp.asarray(x_old_comp, dtype=x_old.dtype)
    dx_old = -(dz * jnp.maximum(width - jnp.abs(x_old), 0) * alpha) * jnp.asarray(x_new_comp, dtype=x_new.dtype)
    return dx_new, dx_old

  return z, grad



