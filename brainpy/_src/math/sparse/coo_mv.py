# -*- coding: utf-8 -*-


from typing import Union, Tuple

from jax import numpy as jnp

from brainpy._src.math.ndarray import BaseArray as Array
import brainevent


__all__ = [
  'coomv',
]


def coomv(
    data: Union[float, jnp.ndarray, Array],
    row: Union[jnp.ndarray, Array],
    col: Union[jnp.ndarray, Array],
    vector: Union[jnp.ndarray, Array],
    *,
    shape: Tuple[int, int],
    rows_sorted: bool = False,
    cols_sorted: bool = False,
    transpose: bool = False,
    method: str = 'cusparse'
):
  """Product of COO sparse matrix and a dense vector using cuSPARSE algorithm.

  This function supports JAX transformations, including `jit()`, `grad()`,
  `vmap()` and `pmap()`.

  Parameters
  ----------
  data: ndarray, float
    An array of shape ``(nse,)``.
  row: ndarray
    An array of shape ``(nse,)``.
  col: ndarray
    An array of shape ``(nse,)`` and dtype ``row.dtype``.
  vector: ndarray
    An array of shape ``(shape[0] if transpose else shape[1],)`` and
    dtype ``data.dtype``.
  shape: tuple of int
    The shape of the sparse matrix.
  rows_sorted: bool
    Row index are sorted.
  cols_sorted: bool
    Column index are sorted.
  transpose: bool
    A boolean specifying whether to transpose the sparse matrix
    before computing.
  method: str
    The method used to compute the matrix-vector multiplication.

  Returns
  -------
  y: ndarray
    An array of shape ``(shape[1] if transpose else shape[0],)`` representing
    the matrix vector product.
  """
  if isinstance(data, Array):
    data = data.value
  if isinstance(row, Array):
    row = row.value
  if isinstance(col, Array):
    col = col.value
  if isinstance(vector, Array):
    vector = vector.value
  csr = brainevent.COO((data, row, col), shape=shape)
  if transpose:
    return vector @ csr
  else:
    return csr @ vector
