# -*- coding: utf-8 -*-


from typing import Union, Optional

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
import numpy as np

from .arrayinterporate import as_jax
from .ndarray import Array

__all__ = [
  'flatten',
  'fill_diagonal',
  'remove_diag',
  'clip_by_norm',
]


def flatten(input: Union[jax.Array, Array],
            start_dim: Optional[int] = None,
            end_dim: Optional[int] = None) -> jax.Array:
  """Flattens input by reshaping it into a one-dimensional tensor.
  If ``start_dim`` or ``end_dim`` are passed, only dimensions starting
  with ``start_dim`` and ending with ``end_dim`` are flattened.
  The order of elements in input is unchanged.

  .. note::
     Flattening a zero-dimensional tensor will return a one-dimensional view.

  Parameters
  ----------
  input: Array
    The input array.
  start_dim: int
    the first dim to flatten
  end_dim: int
    the last dim to flatten

  Returns
  -------
  out: Array
  """
  input = as_jax(input)
  shape = input.shape
  ndim = input.ndim
  if ndim == 0:
    ndim = 1
  if start_dim is None:
    start_dim = 0
  elif start_dim < 0:
    start_dim = ndim + start_dim
  if end_dim is None:
    end_dim = ndim - 1
  elif end_dim < 0:
    end_dim = ndim + end_dim
  end_dim += 1
  if start_dim < 0 or start_dim > ndim:
    raise ValueError(f'start_dim {start_dim} is out of size.')
  if end_dim < 0 or end_dim > ndim:
    raise ValueError(f'end_dim {end_dim} is out of size.')
  new_shape = shape[:start_dim] + (np.prod(shape[start_dim: end_dim], dtype=int), ) + shape[end_dim:]
  return jnp.reshape(input, new_shape)


def fill_diagonal(a, val, inplace=True):
  if a.ndim < 2:
    raise ValueError(f'Only support tensor has dimension >= 2, but got {a.shape}')
  if not isinstance(a, Array) and inplace:
    raise ValueError('``fill_diagonal()`` is used in in-place updating, therefore '
                     'it requires a brainpy Array. If you want to disable '
                     'inplace updating, use ``fill_diagonal(inplace=False)``.')
  val = val.value if isinstance(val, Array) else val
  i, j = jnp.diag_indices(min(a.shape[-2:]))
  r = as_jax(a).at[..., i, j].set(val)
  if inplace:
    a.value = r
  else:
    return r


def remove_diag(arr):
  """Remove the diagonal of the matrix.

  Parameters
  ----------
  arr: ArrayType
    The matrix with the shape of `(M, N)`.

  Returns
  -------
  arr: Array
    The matrix without diagonal which has the shape of `(M, N-1)`.
  """
  if arr.ndim != 2:
    raise ValueError(f'Only support 2D matrix, while we got a {arr.ndim}D array.')
  eyes = Array(jnp.ones(arr.shape, dtype=bool))
  fill_diagonal(eyes, False)
  return jnp.reshape(arr[eyes.value], (arr.shape[0], arr.shape[1] - 1))


def clip_by_norm(t, clip_norm, axis=None):
  def f(l):
    return l * clip_norm / jnp.maximum(jnp.sqrt(jnp.sum(l * l, axis=axis, keepdims=True)), clip_norm)

  return tree_map(f, t)
