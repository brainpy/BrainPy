# -*- coding: utf-8 -*-

"""

Key points for the operator customization:

1. `index` has two kinds of types: int32, int64
2. `data` has two kinds of types: float32, float64
3. `events` has three kinds of types: bool (True or False), float32, float64

"""

from typing import Union, Tuple

import jax

from brainpy._src.dependency_check import import_braintaichi, raise_braintaichi_not_found

bti = import_braintaichi(error_if_not_found=False)

__all__ = [
    'csrmv'
]


def csrmv(
    data: Union[float, jax.Array],
    indices: jax.Array,
    indptr: jax.Array,
    events: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
) -> jax.Array:
    """Product of a sparse CSR matrix and a dense event vector.

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
    events: ndarray
      An array of shape ``(shape[0] if transpose else shape[1],)``
      and dtype ``data.dtype``.
    shape: tuple
      A length-2 tuple representing the matrix shape.
    transpose: bool
      A boolean specifying whether to transpose the sparse matrix
      before computing.
      If ``transpose=True``, the operator will compute based on the
      event-driven property of the ``events`` vector.

    Returns
    -------
    y : Array
      The array of shape ``(shape[1] if transpose else shape[0],)`` representing
      the matrix vector product.
    """
    if bti is None:
        raise_braintaichi_not_found()

    return bti.event_csrmv(data, indices, indptr, events, shape=shape, transpose=transpose)
