# -*- coding: utf-8 -*-

import numpy as np

__all__ = [
  'find_indexes_of_limit_cycle_max',
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
