# -*- coding: utf-8 -*-


from typing import Union, Tuple

from jax import numpy as jnp

from brainpy._src.dependency_check import import_braintaichi, raise_braintaichi_not_found
from brainpy._src.math.ndarray import Array

bti = import_braintaichi(error_if_not_found=False)

__all__ = [
  'csrmv',
]


def csrmv(
    data: Union[float, jnp.ndarray, Array],
    indices: Union[jnp.ndarray, Array],
    indptr: Union[jnp.ndarray, Array],
    vector: Union[jnp.ndarray, Array],
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
):
  """Product of CSR sparse matrix and a dense vector using cuSPARSE algorithm.

  This function supports JAX transformations, including `jit()`, `grad()`,
  `vmap()` and `pmap()`.

  Parameters
  ----------
  data: ndarray, float
    An array of shape ``(nse,)``.
  indices: ndarray
    An array of shape ``(nse,)``.
  indptr: ndarray
    An array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``.
  vector: ndarray
    An array of shape ``(shape[0] if transpose else shape[1],)``
    and dtype ``data.dtype``.
  shape: tuple of int
    A length-2 tuple representing the matrix shape.
  transpose: bool
    A boolean specifying whether to transpose the sparse matrix
    before computing.
  method: str
    The method used to compute Matrix-Vector Multiplication. Default is ``taichi``. 
    The candidate methods are:

    - ``None``: default using Taichi kernel.
    - ``cusparse``: using cuSPARSE library.
    - ``scalar``:
    - ``vector``:
    - ``adaptive``:

  Returns
  -------
  y : ndarry
    The array of shape ``(shape[1] if transpose else shape[0],)`` representing
    the matrix vector product.
  """
  if bti is None:
    raise_braintaichi_not_found()

  return bti.csrmv(data, indices, indptr, vector, shape=shape, transpose=transpose)

