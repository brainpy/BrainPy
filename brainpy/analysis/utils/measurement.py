# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpy as np
from brainpy.tools.others import numba_jit


__all__ = [
  'find_indexes_of_limit_cycle_max',
  'euclidean_distance',
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


# @tools.numba_jit
def euclidean_distance(points: np.ndarray):
  """Get the distance matrix.

  Equivalent to:

  >>> from scipy.spatial.distance import squareform, pdist
  >>> f = lambda points: squareform(pdist(points, metric="euclidean"))

  Parameters
  ----------
  points: jnp.ndarray, bm.JaxArray
    The points.

  Returns
  -------
  dist_matrix: jnp.ndarray
    The distance matrix.
  """
  num_point = points.shape[0]
  indices = np.triu_indices(num_point)
  dist_mat = np.zeros((num_point, num_point))
  for idx in range(len(indices[0])):
    i = indices[0][idx]
    j = indices[1][idx]
    dist_mat[i, j] = np.linalg.norm(points[i] - points[j])
  dist_mat = np.maximum(dist_mat, dist_mat.T)
  return dist_mat

