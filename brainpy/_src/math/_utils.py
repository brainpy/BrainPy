# -*- coding: utf-8 -*-

import functools
from typing import Callable

import jax
from jax.tree_util import tree_map

from .ndarray import Array

__all__ = [
  'npfun_returns_bparray'
]


def _as_jax_array_(obj):
  return obj.value if isinstance(obj, Array) else obj


def _return(a):
  return Array(a) if isinstance(a, jax.Array) and a.ndim > 1 else a


_return_bp_array = True


def npfun_returns_bparray(mode: bool):
  global _return_bp_array
  assert isinstance(mode, bool)
  _return_bp_array = mode


def wraps(fun: Callable):
  """Specialized version of functools.wraps for wrapping numpy functions.

  This produces a wrapped function with a modified docstring. In particular, if
  `update_doc` is True, parameters listed in the wrapped function that are not
  supported by the decorated function will be removed from the docstring. For
  this reason, it is important that parameter names match those in the original
  numpy function.
  """

  def wrap(op):
    docstr = getattr(fun, "__doc__", None)
    op.__doc__ = docstr
    op.__wrapped__ = fun
    return op

  return wrap


def _is_leaf(a):
  return isinstance(a, Array)


def _compatible_with_brainpy_array(fun: Callable):
  @functools.wraps(fun)
  def new_fun(*args, **kwargs):
    args = tree_map(_as_jax_array_, args, is_leaf=_is_leaf)
    if len(kwargs):
      kwargs = tree_map(_as_jax_array_, kwargs, is_leaf=_is_leaf)
    r = fun(*args, **kwargs)
    return tree_map(_return, r) if _return_bp_array else r

  new_fun.__doc__ = getattr(fun, "__doc__", None)

  return new_fun
