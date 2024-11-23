# -*- coding: utf-8 -*-


from typing import Union, Tuple

from jax import numpy as jnp

from brainpy._src.math.ndarray import Array
from brainpy._src.dependency_check import import_braintaichi, raise_braintaichi_not_found

bti = import_braintaichi(error_if_not_found=False)

__all__ = [
  'csrmm',
]


def csrmm(
    data: Union[float, jnp.ndarray, Array],
    indices: Union[jnp.ndarray, Array],
    indptr: Union[jnp.ndarray, Array],
    matrix: Union[jnp.ndarray, Array],
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
):
  """
  Product of CSR sparse matrix and a dense matrix.

  Args:
      data : array of shape ``(nse,)``.
      indices : array of shape ``(nse,)``
      indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
      B : array of shape ``(shape[0] if transpose else shape[1], cols)`` and
      dtype ``data.dtype``
      shape : length-2 tuple representing the matrix shape
      transpose : boolean specifying whether to transpose the sparse matrix
      before computing.

  Returns:
      C : array of shape ``(shape[1] if transpose else shape[0], cols)``
      representing the matrix-matrix product.
  """
  if bti is None:
    raise_braintaichi_not_found()

  return bti.csrmm(data, indices, indptr, matrix, shape=shape, transpose=transpose)