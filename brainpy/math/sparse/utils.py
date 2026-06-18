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
    """convert pre_ids, post_ids to (indices, indptr)."""
    pre_ids = as_jax(pre_ids)
    post_ids = as_jax(post_ids)

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
