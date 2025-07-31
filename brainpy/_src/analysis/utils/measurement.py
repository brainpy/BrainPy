# -*- coding: utf-8 -*-

from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten

import brainpy._src.math as bm
from brainpy.tools import numba_jit

__all__ = [
  'find_indexes_of_limit_cycle_max',
  'euclidean_distance',
  'euclidean_distance_jax',
]


@numba_jit
def _f1(arr, grad, tol):
  condition = np.logical_and(grad[:-1] * grad[1:] <= 0, grad[:-1] >= 0)
  indexes = np.where(condition)[0]
  if len(indexes) >= 2:
    data = arr[indexes[-2]: indexes[-1]]
    length = np.max(data) - np.min(data)
    a = arr[indexes[-2]]
    b = arr[indexes[-1]]
    # TODO: how to choose length threshold, 1e-3?
    if length > 1e-3 and np.abs(a - b) <= tol * length:
      return indexes[-2:]
  return np.array([-1, -1])


def find_indexes_of_limit_cycle_max(arr, tol=0.001):
  grad = np.gradient(arr)
  return _f1(arr, grad, tol)


@numba_jit
def euclidean_distance(points: np.ndarray, num_point=None):
  """Get the distance matrix.

  Equivalent to:

  >>> from scipy.spatial.distance import squareform, pdist
  >>> f = lambda points: squareform(pdist(points, metric="euclidean"))

  Parameters
  ----------
  points: ArrayType
    The points.

  Returns
  -------
  dist_matrix: jnp.ndarray
    The distance matrix.
  """

  if isinstance(points, dict):
    if num_point is None:
      raise ValueError('Please provide num_point')
    indices = np.triu_indices(num_point)
    dist_mat = np.zeros((num_point, num_point))
    for idx in range(len(indices[0])):
      i = indices[0][idx]
      j = indices[1][idx]
      dist_mat[i, j] = np.sqrt(np.sum([np.sum((value[i] - value[j]) ** 2) for value in points.values()]))
  else:
    num_point = points.shape[0]
    indices = np.triu_indices(num_point)
    dist_mat = np.zeros((num_point, num_point))
    for idx in range(len(indices[0])):
      i = indices[0][idx]
      j = indices[1][idx]
      dist_mat[i, j] = np.linalg.norm(points[i] - points[j])
  dist_mat = np.maximum(dist_mat, dist_mat.T)
  return dist_mat


@jax.jit
@partial(jax.vmap, in_axes=[0, 0, None])
def _ed(i, j, leaves):
  squares = jnp.asarray([((leaf[i] - leaf[j]) ** 2).sum() for leaf in leaves])
  return jnp.sqrt(jnp.sum(squares))


def euclidean_distance_jax(points: Union[jnp.ndarray, bm.ndarray], num_point=None):
  """Get the distance matrix.

  Equivalent to:

  >>> from scipy.spatial.distance import squareform, pdist
  >>> f = lambda points: squareform(pdist(points, metric="euclidean"))

  Parameters
  ----------
  points: ArrayType
    The points.
  num_point: int

  Returns
  -------
  dist_matrix: ArrayType
    The distance matrix.
  """
  if isinstance(points, dict):
    if num_point is None:
      raise ValueError('Please provide num_point')
  else:
    num_point = points.shape[0]
  indices = jnp.triu_indices(num_point)
  dist_mat = bm.zeros((num_point, num_point))
  leaves, _ = tree_flatten(points)
  dist_mat[indices] = _ed(*indices, leaves)
  dist_mat = jnp.maximum(dist_mat.value, dist_mat.value.T)
  return dist_mat

