# -*- coding: utf-8 -*-

import functools
from typing import Callable

from jax.tree_util import tree_map

from .ndarray import Array, _return


def _as_jax_array_(obj):
  return obj.value if isinstance(obj, Array) else obj


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
    out = None
    if len(kwargs):
      # compatible with PyTorch syntax
      if 'dim' in kwargs:
        kwargs['axis'] = kwargs.pop('dim')
      if 'keepdim' in kwargs:
        kwargs['keepdims'] = kwargs.pop('keepdim')
      # compatible with TensorFlow syntax
      if 'keep_dims' in kwargs:
        kwargs['keepdims'] = kwargs.pop('keep_dims')
      # compatible with NumPy/PyTorch syntax
      if 'out' in kwargs:
        out = kwargs.pop('out')
        if not isinstance(out, Array):
          raise TypeError(f'"out" must be an instance of brainpy Array. While we got {type(out)}')
      # format
      kwargs = tree_map(_as_jax_array_, kwargs, is_leaf=_is_leaf)
    r = fun(*args, **kwargs)
    if out is None:
      return tree_map(_return, r)
    else:
      out.value = r

  new_fun.__doc__ = getattr(fun, "__doc__", None)

  return new_fun
