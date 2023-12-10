# -*- coding: utf-8 -*-


from typing import Optional

import jax.numpy as jnp
from jax.tree_util import tree_map

from brainpy import check, tools
from .compat_numpy import fill_diagonal
from .environment import get_dt, get_int
from .ndarray import Array
from .interoperability import as_jax

__all__ = [
  'shared_args_over_time',
  'remove_diag',
  'clip_by_norm',
  'exprel',
]


def shared_args_over_time(num_step: Optional[int] = None,
                          duration: Optional[float] = None,
                          dt: Optional[float] = None,
                          t0: float = 0.,
                          include_dt: bool = True):
  """Form a shared argument over time for the inference of a :py:class:`~.DynamicalSystem`.

  Parameters
  ----------
  num_step: int
    The number of time step. Provide either ``duration`` or ``num_step``.
  duration: float
    The total duration. Provide either ``duration`` or ``num_step``.
  dt: float
    The duration for each time step.
  t0: float
    The start time.
  include_dt: bool
    Produce the time steps at every time step.

  Returns
  -------
  shared: DotDict
    The shared arguments over the given time.
  """
  dt = get_dt() if dt is None else dt
  check.is_float(dt, 'dt', allow_none=False)
  if duration is None:
    check.is_integer(num_step, 'num_step', allow_none=False)
  else:
    check.is_float(duration, 'duration', allow_none=False)
    num_step = int(duration / dt)
  r = tools.DotDict(i=jnp.arange(num_step, dtype=get_int()))
  r['t'] = r['i'] * dt + t0
  if include_dt:
    r['dt'] = jnp.ones_like(r['t']) * dt
  return r


def remove_diag(arr):
  """Remove the diagonal of the matrix.

  Parameters
  ----------
  arr: ArrayType
    The matrix with the shape of `(M, N)`.

  Returns
  -------
  arr: Array
    The matrix without diagonal which has the shape of `(M, N-1)`.
  """
  if arr.ndim != 2:
    raise ValueError(f'Only support 2D matrix, while we got a {arr.ndim}D array.')
  eyes = Array(jnp.ones(arr.shape, dtype=bool))
  fill_diagonal(eyes, False)
  return jnp.reshape(arr[eyes.value], (arr.shape[0], arr.shape[1] - 1))


def clip_by_norm(t, clip_norm, axis=None):
  def f(l):
    return l * clip_norm / jnp.maximum(jnp.sqrt(jnp.sum(l * l, axis=axis, keepdims=True)), clip_norm)

  return tree_map(f, t)


def _exprel(x, threshold):
  def true_f(x):
    x2 = x * x
    return 1. + x / 2. + x2 / 6. + x2 * x / 24.0  # + x2 * x2 / 120.

  def false_f(x):
    return (jnp.exp(x) - 1) / x

  # return jax.lax.cond(jnp.abs(x) < threshold, true_f, false_f, x)
  return jnp.where(jnp.abs(x) <= threshold, 1. + x / 2. + x * x / 6., (jnp.exp(x) - 1) / x)


def exprel(x, threshold: float = None):
  """Relative error exponential, ``(exp(x) - 1)/x``.

  When ``x`` is near zero, ``exp(x)`` is near 1, so the numerical calculation of ``exp(x) - 1`` can
  suffer from catastrophic loss of precision. ``exprel(x)`` is implemented to avoid the loss of
  precision that occurs when ``x`` is near zero.

  Args:
    x: ndarray. Input array. ``x`` must contain real numbers.
    threshold: float.

  Returns:
    ``(exp(x) - 1)/x``, computed element-wise.
  """
  x = as_jax(x)
  if threshold is None:
    if hasattr(x, 'dtype') and x.dtype == jnp.float64:
      threshold = 1e-8
    else:
      threshold = 1e-5
  return _exprel(x, threshold)
