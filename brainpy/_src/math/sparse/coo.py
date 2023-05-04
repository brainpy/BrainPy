# -*- coding: utf-8 -*-


from typing import Tuple
from typing import Union

import jax.numpy as jnp

from brainpy._src import tools
from brainpy._src.math.ndarray import Array

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
    transpose: bool = False
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

  Returns
  -------
  y: ndarray
    An array of shape ``(shape[1] if transpose else shape[0],)`` representing
    the matrix vector product.
  """
  bl = tools.import_brainpylib()
  return bl.sparse_ops.cusparse_coo_matvec(data,
                                           row,
                                           col,
                                           vector,
                                           shape=shape,
                                           rows_sorted=rows_sorted,
                                           cols_sorted=cols_sorted,
                                           transpose=transpose)

