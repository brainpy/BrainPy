# -*- coding: utf-8 -*-


import jax.numpy as jnp
from jax.core import Primitive, ShapedArray
from jax.interpreters import batching, mlir, xla
from .ndarray import Array

__all__ = [
  'remove_vmap'
]


def remove_vmap(x, op='any'):
  if isinstance(x, Array):
    x = x.value
  if op == 'any':
    return _any_without_vmap(x)
  elif op == 'all':
    return _all_without_vmap(x)
  else:
    raise ValueError(f'Do not support type: {op}')


_any_no_vmap_prim = Primitive('any_no_vmap')


def _any_without_vmap(x):
  return _any_no_vmap_prim.bind(x)


def _any_without_vmap_imp(x):
  return jnp.any(x)


def _any_without_vmap_abs(x):
  return ShapedArray(shape=(), dtype=jnp.bool_)


def _any_without_vmap_batch(x, batch_axes):
  (x, ) = x
  return _any_without_vmap(x), batching.not_mapped


_any_no_vmap_prim.def_impl(_any_without_vmap_imp)
_any_no_vmap_prim.def_abstract_eval(_any_without_vmap_abs)
batching.primitive_batchers[_any_no_vmap_prim] = _any_without_vmap_batch
if hasattr(xla, "lower_fun"):
    xla.register_translation(_any_no_vmap_prim,
                             xla.lower_fun(_any_without_vmap_imp, multiple_results=False, new_style=True))
mlir.register_lowering(_any_no_vmap_prim, mlir.lower_fun(_any_without_vmap_imp, multiple_results=False))


_all_no_vmap_prim = Primitive('all_no_vmap')


def _all_without_vmap(x):
  return _all_no_vmap_prim.bind(x)


def _all_without_vmap_imp(x):
  return jnp.all(x)


def _all_without_vmap_abs(x):
  return ShapedArray(shape=(), dtype=jnp.bool_)


def _all_without_vmap_batch(x, batch_axes):
  (x, ) = x
  return _all_without_vmap(x), batching.not_mapped


_all_no_vmap_prim.def_impl(_all_without_vmap_imp)
_all_no_vmap_prim.def_abstract_eval(_all_without_vmap_abs)
batching.primitive_batchers[_all_no_vmap_prim] = _all_without_vmap_batch
if hasattr(xla, "lower_fun"):
    xla.register_translation(_all_no_vmap_prim,
                             xla.lower_fun(_all_without_vmap_imp, multiple_results=False, new_style=True))
mlir.register_lowering(_all_no_vmap_prim, mlir.lower_fun(_all_without_vmap_imp, multiple_results=False))

