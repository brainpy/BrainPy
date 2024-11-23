# -*- coding: utf-8 -*-


from typing import Union, Tuple

from jax import numpy as jnp

from brainpy._src.math.ndarray import Array
from brainpy._src.dependency_check import import_braintaichi, raise_braintaichi_not_found

bti = import_braintaichi(error_if_not_found=False)

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
  if bti is None:
    raise_braintaichi_not_found()

  return bti.coomv(
    data=data,
    row=row,
    col=col,
    vector=vector,
    shape=shape,
    rows_sorted=rows_sorted,
    cols_sorted=cols_sorted,
    transpose=transpose,
    method=method
  )
