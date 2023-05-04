# -*- coding: utf-8 -*-


from typing import Tuple

import jax.numpy as jnp

from brainpy._src import tools


__all__ = [
  'coo_to_csr',
  'csr_to_coo',
  'csr_to_dense'
]


def coo_to_csr(
    pre_ids: jnp.ndarray,
    post_ids: jnp.ndarray,
    *,
    num_row: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """convert pre_ids, post_ids to (indices, indptr)."""
  bl = tools.import_brainpylib()
  return bl.sparse_ops.coo_to_csr(pre_ids, post_ids, num_row=num_row)


def csr_to_coo(
    indices: jnp.ndarray,
    indptr: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Given CSR (indices, indptr) return COO (row, col)"""
  bl = tools.import_brainpylib()
  return bl.sparse_ops.csr_to_coo(indices, indptr)


def csr_to_csc():
  pass


def coo_to_dense(
    data: jnp.ndarray,
    rows: jnp.ndarray,
    cols: jnp.ndarray,
    *,
    shape: Tuple[int, int]
) -> jnp.ndarray:
  pass


def csr_to_dense(
    data: jnp.ndarray,
    indices: jnp.ndarray,
    indptr: jnp.ndarray,
    *,
    shape: Tuple[int, int]
) -> jnp.ndarray:
  bl = tools.import_brainpylib()
  return bl.sparse_ops.csr_to_dense(data, indices, indptr, shape=shape)
