# -*- coding: utf-8 -*-


from jax.tree_util import tree_flatten

import brainpy.math as bm
from brainpy.errors import UnsupportedError

_reduction_error = 'Only support reduction of "mean", "sum" and "none", but we got "%s".'


def _is_leaf(x):
  return isinstance(x, bm.Array)


def _reduce(outputs, reduction, axis=None):
  if reduction == 'mean':
    return outputs.mean(axis)
  elif reduction == 'sum':
    return outputs.sum(axis)
  elif reduction == 'none':
    return outputs
  else:
    raise UnsupportedError(_reduction_error % reduction)


def _multi_return(r):
  leaves = tree_flatten(r)[0]
  r = leaves[0]
  for leaf in leaves[1:]:
    r += leaf
  return r
