# -*- coding: utf-8 -*-

import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten

from ._utils import wraps
from .ndarray import Array

__all__ = [
  'empty', 'empty_like', 'ones', 'ones_like', 'zeros', 'zeros_like',
  'array', 'asarray', 'arange', 'linspace', 'logspace',
]


def _as_jax_array_(obj):
  return obj.value if isinstance(obj, Array) else obj


@wraps(jnp.zeros)
def zeros(shape, dtype=None):
  return Array(jnp.zeros(shape, dtype=dtype))


@wraps(jnp.ones)
def ones(shape, dtype=None):
  return Array(jnp.ones(shape, dtype=dtype))


@wraps(jnp.empty)
def empty(shape, dtype=None):
  return Array(jnp.zeros(shape, dtype=dtype))


@wraps(jnp.zeros_like)
def zeros_like(a, dtype=None, shape=None):
  a = _as_jax_array_(a)
  return Array(jnp.zeros_like(a, dtype=dtype, shape=shape))


@wraps(jnp.ones_like)
def ones_like(a, dtype=None, shape=None):
  a = _as_jax_array_(a)
  return Array(jnp.ones_like(a, dtype=dtype, shape=shape))


@wraps(jnp.empty_like)
def empty_like(a, dtype=None, shape=None):
  a = _as_jax_array_(a)
  return Array(jnp.zeros_like(a, dtype=dtype, shape=shape))


@wraps(jnp.array)
def array(a, dtype=None, copy=True, order="K", ndmin=0) -> Array:
  a = _as_jax_array_(a)
  try:
    res = jnp.array(a, dtype=dtype, copy=copy, order=order, ndmin=ndmin)
  except TypeError:
    leaves, tree = tree_flatten(a, is_leaf=lambda a: isinstance(a, Array))
    leaves = [_as_jax_array_(l) for l in leaves]
    a = tree_unflatten(tree, leaves)
    res = jnp.array(a, dtype=dtype, copy=copy, order=order, ndmin=ndmin)
  return Array(res)


@wraps(jnp.asarray)
def asarray(a, dtype=None, order=None):
  a = _as_jax_array_(a)
  try:
    res = jnp.asarray(a=a, dtype=dtype, order=order)
  except TypeError:
    leaves, tree = tree_flatten(a, is_leaf=lambda a: isinstance(a, Array))
    leaves = [_as_jax_array_(l) for l in leaves]
    arrays = tree_unflatten(tree, leaves)
    res = jnp.asarray(a=arrays, dtype=dtype, order=order)
  return Array(res)


@wraps(jnp.arange)
def arange(*args, **kwargs):
  args = [_as_jax_array_(a) for a in args]
  kwargs = {k: _as_jax_array_(v) for k, v in kwargs.items()}
  return Array(jnp.arange(*args, **kwargs))


@wraps(jnp.linspace)
def linspace(*args, **kwargs):
  args = [_as_jax_array_(a) for a in args]
  kwargs = {k: _as_jax_array_(v) for k, v in kwargs.items()}
  res = jnp.linspace(*args, **kwargs)
  if isinstance(res, tuple):
    return Array(res[0]), res[1]
  else:
    return Array(res)


@wraps(jnp.logspace)
def logspace(*args, **kwargs):
  args = [_as_jax_array_(a) for a in args]
  kwargs = {k: _as_jax_array_(v) for k, v in kwargs.items()}
  return Array(jnp.logspace(*args, **kwargs))

