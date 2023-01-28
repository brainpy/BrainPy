from typing import Union, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .ndarray import Array, _as_jax_array_

__all__ = [
  'flatten',
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
  input = _as_jax_array_(input)
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

