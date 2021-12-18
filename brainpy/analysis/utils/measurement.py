# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpy as np
from jax import jit, vmap

import brainpy.math as bm

__all__ = [
  'find_indexes_of_limit_cycle_max',
  'euclidean_distance',
]


def _f1(arr, grad, tol):
  condition = np.logical_and(grad[:-1] * grad[1:] <= 0, grad[:-1] >= 0)
  indexes = np.where(condition)[0]
  if len(indexes) >= 2:
    data = arr[indexes[-2]: indexes[-1]]
    length = np.max(data) - np.min(data)
    a = arr[indexes[-2]]
    b = arr[indexes[-1]]
    if np.abs(a - b) < tol * length:
      return indexes[-2:]
  return np.array([-1, -1])


def find_indexes_of_limit_cycle_max(arr, tol=0.001):
  grad = np.gradient(arr)
  return _f1(arr, grad, tol)


def euclidean_distance(points: jnp.ndarray):
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
  points = points.value if isinstance(points, bm.JaxArray) else points
  num_point = points.shape[0]
  indices = jnp.triu_indices(num_point)
  f = jit(vmap(lambda i, j: jnp.linalg.norm(points[i] - points[j])))
  dists = f(*indices)
  dist_mat = bm.zeros((num_point, num_point))
  dist_mat[indices] = dists
  return dist_mat

