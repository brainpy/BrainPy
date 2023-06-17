# -*- coding: utf-8 -*-

from jax.tree_util import tree_flatten, tree_map

import jax.numpy as jnp
import brainpy.math as bm
from .utils import _is_leaf, _multi_return

__all__ = [
  'l2_norm',
  'mean_absolute',
  'mean_square',
  'log_cosh',
  'smooth_labels',
]


def l2_norm(x, axis=None):
  """Computes the L2 loss.

  Args:
      x: n-dimensional tensor of floats.

  Returns:
      scalar tensor containing the l2 loss of x.
  """
  leaves, _ = tree_flatten(x)
  return jnp.sqrt(jnp.sum(jnp.asarray([jnp.vdot(x, x) for x in leaves]), axis=axis))


def mean_absolute(outputs, axis=None):
  r"""Computes the mean absolute error between x and y.

  Returns:
      tensor of shape (d_i, ..., for i in keep_axis) containing the mean absolute error.
  """
  r = tree_map(lambda a: bm.mean(bm.abs(a), axis=axis), outputs, is_leaf=_is_leaf)
  return _multi_return(r)


def mean_square(predicts, axis=None):
  r = tree_map(lambda a: bm.mean(a ** 2, axis=axis), predicts, is_leaf=_is_leaf)
  return _multi_return(r)


def log_cosh(errors):
  r"""Calculates the log-cosh loss for a set of predictions.

  log(cosh(x)) is approximately `(x**2) / 2` for small x and `abs(x) - log(2)`
  for large x.  It is a twice differentiable alternative to the Huber loss.
  References:
    [Chen et al, 2019](https://openreview.net/pdf?id=rkglvsC9Ym)
  Args:
    errors: a vector of arbitrary shape.
  Returns:
    the log-cosh loss.
  """
  r = tree_map(lambda a: bm.logaddexp(a, -a) - bm.log(2.0).astype(a.dtype),
               errors, is_leaf=_is_leaf)
  return _multi_return(r)


def smooth_labels(labels, alpha: float) -> jnp.ndarray:
  r"""Apply label smoothing.
  Label smoothing is often used in combination with a cross-entropy loss.
  Smoothed labels favour small logit gaps, and it has been shown that this can
  provide better model calibration by preventing overconfident predictions.
  References:
    [Müller et al, 2019](https://arxiv.org/pdf/1906.02629.pdf)
  Args:
    labels: one hot labels to be smoothed.
    alpha: the smoothing factor, the greedy category with be assigned
      probability `(1-alpha) + alpha / num_categories`
  Returns:
    a smoothed version of the one hot input labels.
  """
  r = tree_map(lambda tar: (1.0 - alpha) * tar + alpha / tar.shape[-1],
               labels, is_leaf=lambda x: isinstance(x, bm.Array))
  return _multi_return(r)
