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
from typing import Union, Tuple

import brainevent
from jax import numpy as jnp

from brainpy.math.ndarray import Array as Array

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

    Parameters::

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

    Returns::

    y : ndarry
      The array of shape ``(shape[1] if transpose else shape[0],)`` representing
      the matrix vector product.
    """
    if isinstance(data, Array):
        data = data.value
    if isinstance(indices, Array):
        indices = indices.value
    if isinstance(indptr, Array):
        indptr = indptr.value
    if isinstance(vector, Array):
        vector = vector.value
    csr = brainevent.CSR((data, indices, indptr), shape=shape)
    if transpose:
        return vector @ csr
    else:
        return csr @ vector
