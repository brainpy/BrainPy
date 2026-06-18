# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Tuple

import brainevent
from jax import numpy as jnp

from brainpy.math.interoperability import as_jax

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
    """Convert COO ``(pre_ids, post_ids)`` connectivity to CSR ``(indices, indptr)``.

    Parameters
    ----------
    pre_ids : ndarray
        Row (pre-synaptic) index of each non-zero entry. Every value must be in
        ``[0, num_row)``.
    post_ids : ndarray
        Column (post-synaptic) index of each non-zero entry, aligned with
        ``pre_ids``.
    num_row : int
        Number of rows of the sparse matrix (``shape[0]``).

    Returns
    -------
    indices : ndarray
        CSR column indices of shape ``(nse,)``.
    indptr : ndarray
        CSR row pointers of shape ``(num_row + 1,)`` and dtype ``int32``.

    Raises
    ------
    ValueError
        If any ``pre_ids`` falls outside ``[0, num_row)``. Such an entry would
        otherwise be silently dropped from ``indptr`` (its scatter index is
        out-of-bounds), producing a structurally invalid CSR in which
        ``indptr[-1] != len(indices)``.

    Notes
    -----
    This is an eager preprocessing helper: it relies on ``jnp.unique`` (whose
    output size is data-dependent) and therefore cannot be traced under
    ``jit``/``vmap``.
    """
    pre_ids = as_jax(pre_ids)
    post_ids = as_jax(post_ids)

    # Validate the pre (row) indices eagerly. An out-of-range ``pre_id`` would be
    # silently dropped by the out-of-bounds ``.at[].set`` scatter below, yielding
    # a corrupt CSR (``indptr[-1] != nse``) instead of an error. ``coo_to_csr``
    # already cannot be ``jit``-traced (``jnp.unique``), so this concrete check
    # does not regress any JAX transformation behaviour.
    if pre_ids.size > 0:
        pre_min = int(jnp.min(pre_ids))
        pre_max = int(jnp.max(pre_ids))
        if pre_min < 0 or pre_max >= num_row:
            raise ValueError(
                f'"pre_ids" must lie in [0, num_row) = [0, {num_row}), '
                f'but got values in [{pre_min}, {pre_max}].'
            )

    # sorting
    sort_ids = jnp.argsort(pre_ids, stable=True)
    post_ids = post_ids[sort_ids]

    indices = post_ids
    unique_pre_ids, pre_count = jnp.unique(pre_ids, return_counts=True)
    final_pre_count = jnp.zeros(num_row, dtype=jnp.int32)
    final_pre_count = final_pre_count.at[unique_pre_ids].set(pre_count)
    indptr = final_pre_count.cumsum()
    indptr = jnp.insert(indptr, 0, 0).astype(jnp.int32)
    return indices, indptr


def csr_to_coo(
    indices: jnp.ndarray,
    indptr: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Given CSR (indices, indptr) return COO (row, col)"""
    indices = as_jax(indices)
    indptr = as_jax(indptr)
    return jnp.cumsum(jnp.zeros_like(indices).at[indptr].add(1)) - 1, indices


def csr_to_dense(data, indices, indptr, *, shape):
    """Convert a CSR sparse matrix to a dense array.

    Parameters
    ----------
    data : ndarray
        An array of shape ``(nse,)`` holding the non-zero values.
    indices : ndarray
        An array of shape ``(nse,)`` holding the column index of each value.
    indptr : ndarray
        An array of shape ``(shape[0] + 1,)`` holding the row pointers.
    shape : tuple of int
        A length-2 tuple ``(n_rows, n_cols)`` for the dense matrix.

    Returns
    -------
    dense : ndarray
        The dense matrix of shape ``shape``.
    """
    return brainevent.CSR((data, indices, indptr), shape=shape).todense()
