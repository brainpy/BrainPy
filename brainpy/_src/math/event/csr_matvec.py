# -*- coding: utf-8 -*-

"""

Key points for the operator customization:

1. `index` has two kinds of types: int32, int64
2. `data` has two kinds of types: float32, float64
3. `events` has three kinds of types: bool (True or False), float32, float64

"""

from typing import Union, Tuple

import brainevent
import jax

from brainpy._src.math.ndarray import BaseArray as Array

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

    Parameters::

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

    Returns::

    y : Array
      The array of shape ``(shape[1] if transpose else shape[0],)`` representing
      the matrix vector product.
    """

    if isinstance(data, Array):
        data = data.value
    if isinstance(indices, Array):
        indices = indices.value
    if isinstance(indptr, Array):
        indptr = indptr.value
    if isinstance(events, Array):
        events = events.value

    events = brainevent.EventArray(events)
    csr = brainevent.CSR((data, indices, indptr), shape=shape)
    if transpose:
        return events @ csr
    else:
        return csr @ events
