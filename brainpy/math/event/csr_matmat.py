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
    """Product of CSR sparse matrix and a dense event matrix.

    Args:
        data : array of shape ``(nse,)``, float.
        indices : array of shape ``(nse,)``
        indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
        B : array of shape ``(shape[0] if transpose else shape[1], cols)`` and
        dtype ``data.dtype``
        shape : length-2 tuple representing the matrix shape
        transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
        C : array of shape ``(shape[1] if transpose else shape[0], cols)``
        representing the matrix-matrix product product.
    """
    if isinstance(data, Array):
        data = data.value
    if isinstance(indices, Array):
        indices = indices.value
    if isinstance(indptr, Array):
        indptr = indptr.value
    if isinstance(matrix, Array):
        matrix = matrix.value

    matrix = brainevent.EventArray(matrix)
    csr = brainevent.CSR((data, indices, indptr), shape=shape)
    if transpose:
        return matrix @ csr
    else:
        return csr @ matrix
