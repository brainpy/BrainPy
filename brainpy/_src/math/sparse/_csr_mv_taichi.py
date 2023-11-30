# -*- coding: utf-8 -*-


from functools import partial
from typing import Union, Tuple

import jax
import numba
import numpy as np
import taichi as ti
from jax import core, dtypes
from jax import numpy as jnp
from jax.interpreters import ad, mlir, xla
from jax.lib import xla_client
from jaxlib import gpu_sparse

from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import Array
from brainpy._src.math.op_register import XLACustomOp
from brainpy._src.math.sparse._utils import csr_to_coo
from brainpy._src.dependency_check import import_brainpylib_gpu_ops
from brainpy.errors import GPUOperatorNotFound

__all__ = [
    'csrmv_taichi',
]

_event_csr_matvec_p = None

@ti.kernel
def _sparse_csr_matvec_cpu_transpose(values: ti.types.ndarray(ndim=1), 
                                     col_indices: ti.types.ndarray(ndim=1), 
                                     row_ptr: ti.types.ndarray(ndim=1), 
                                     vector: ti.types.ndarray(ndim=1), 
                                     shape: ti.types.ndarray(ndim=1),
                                     transpose: ti.types.ndarray(ndim=1),
                                     out: ti.types.ndarray(ndim=1)):
    if values.shape[0] == 1:
        value = values[0]
        ti.loop_config(serialize=True)
        for row_i in range(shape[0]):
            for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
                out[col_indices[j]] += value * vector[row_i]
    
    else:
        ti.loop_config(serialize=True)
        for row_i in range(shape[0]):
            for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
                out[col_indices[j]] += values[j] * vector[row_i]

@ti.kernel
def _sparse_csr_matvec_cpu(values: ti.types.ndarray(ndim=1), 
                           col_indices: ti.types.ndarray(ndim=1), 
                           row_ptr: ti.types.ndarray(ndim=1), 
                           vector: ti.types.ndarray(ndim=1), 
                           shape: ti.types.ndarray(ndim=1),
                           transpose: ti.types.ndarray(ndim=1),
                           out: ti.types.ndarray(ndim=1)):
    if values.shape[0] == 1:
        value = values[0]
        ti.loop_config(serialize=True)
        for row_i in range(shape[0]):
            r = 0.
            for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
                r += value * vector[col_indices[j]]
            out[row_i] = r
    
    else:
        ti.loop_config(serialize=True)
        for row_i in range(shape[0]):
            r = 0.
            for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
                r += values[j] * vector[col_indices[j]]
            out[row_i] = r


@ti.kernel
def _sparse_csr_matvec_gpu_transpose(values: ti.types.ndarray(ndim=1), 
                                     col_indices: ti.types.ndarray(ndim=1), 
                                     row_ptr: ti.types.ndarray(ndim=1), 
                                     vector: ti.types.ndarray(ndim=1), 
                                     shape: ti.types.ndarray(ndim=1),
                                     transpose: ti.types.ndarray(ndim=1),
                                     out: ti.types.ndarray(ndim=1)):
    if values.shape[0] == 1:
        value = values[0]
        for row_i in range(shape[0]):
            for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
                out[col_indices[j]] += value * vector[row_i]
    
    else:
        for row_i in range(shape[0]):
            for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
                out[col_indices[j]] += values[j] * vector[row_i]

@ti.kernel
def _sparse_csr_matvec_gpu(values: ti.types.ndarray(ndim=1), 
                           col_indices: ti.types.ndarray(ndim=1), 
                           row_ptr: ti.types.ndarray(ndim=1), 
                           vector: ti.types.ndarray(ndim=1), 
                           shape: ti.types.ndarray(ndim=1),
                           transpose: ti.types.ndarray(ndim=1),
                           out: ti.types.ndarray(ndim=1)):
    if values.shape[0] == 1:
        value = values[0]
        for row_i in range(shape[0]):
            r = 0.
            for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
                r += value * vector[col_indices[j]]
            out[row_i] = r
    
    else:
        for row_i in range(shape[0]):
            r = 0.
            for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
                r += values[j] * vector[col_indices[j]]
            out[row_i] = r


def csrmv_taichi(
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

    Returns
    -------
    y : ndarry
      The array of shape ``(shape[1] if transpose else shape[0],)`` representing
      the matrix vector product.
    """

    data = jnp.atleast_1d(as_jax(data))
    indices = as_jax(indices)
    indptr = as_jax(indptr)
    vector = as_jax(vector)

    if vector.dtype == jnp.bool_:
        vector = as_jax(vector, dtype=data.dtype)

    if data.dtype not in [jnp.float16, jnp.float32, jnp.float64]:
        raise TypeError('Only support float16, float32 or float64 type. '
                        f'But we got {data.dtype}.')
    if data.dtype != vector.dtype:
        raise TypeError('The types of data and vector should be the same. '
                        f'But we got {data.dtype} != {vector.dtype}.')
    assert data.ndim == indices.ndim == indptr.ndim == vector.ndim == 1
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise ValueError('indices should be a 1D vector with integer type.')
    if not jnp.issubdtype(indptr.dtype, jnp.integer):
        raise ValueError('indptr should be a 1D vector with integer type.')
    out_shape = shape[1] if transpose else shape[0]

    global _event_csr_matvec_p
    if transpose:
        _event_csr_matvec_p = XLACustomOp(cpu_kernel=_sparse_csr_matvec_cpu_transpose,
                                         gpu_kernel=_sparse_csr_matvec_gpu_transpose)
    else:
        _event_csr_matvec_p = XLACustomOp(cpu_kernel=_sparse_csr_matvec_cpu, 
                                         gpu_kernel=_sparse_csr_matvec_gpu)

    shape_list = jnp.array(shape)
    is_transpose = jnp.array(transpose)

    return _event_csr_matvec_p(data,
                               indices,
                               indptr,
                               vector,
                               shape_list,
                               is_transpose,
                               outs=[jax.ShapeDtypeStruct((out_shape,), dtype=data.dtype)]
                               )