# -*- coding: utf-8 -*-


from typing import Union, Tuple

import jax.numpy as jnp

from brainpy._src import tools

__all__ = [
  'event_csr_matvec', 'event_info'
]


def event_csr_matvec(
    data: Union[float, jnp.ndarray],
    indices: jnp.ndarray,
    indptr: jnp.ndarray,
    events: jnp.ndarray,
    *,
    shape: Tuple[int, int],
    transpose: bool = False
) -> jnp.ndarray:
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

  Returns
  -------
  y : ndarry
    The array of shape ``(shape[1] if transpose else shape[0],)`` representing
    the matrix vector product.
  """
  bl = tools.import_brainpylib()
  # computing
  return bl.event_ops.event_csr_matvec(data, indices, indptr, events, shape=shape, transpose=transpose)


def event_info(events: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Collect event information, including event indices, and event number.

  This function supports JAX transformations, including `jit()`,
  `vmap()` and `pmap()`.

  Parameters
  ----------
  events: jnp.ndarray
    The events.

  Returns
  -------
  res: tuple
    A tuple with two elements, denoting the event indices and the event number.
  """
  bl = tools.import_brainpylib()
  return bl.event_ops.event_info(events)

