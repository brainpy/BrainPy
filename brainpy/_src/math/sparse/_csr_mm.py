# -*- coding: utf-8 -*-


from functools import partial
from typing import Union, Tuple

import jax
import numba
import numpy as np
from jax import core, dtypes
from jax import numpy as jnp
from jax.interpreters import ad, mlir, xla
from jax.lib import xla_client
from jaxlib import gpu_sparse

from brainpy._src.dependency_check import import_brainpylib_gpu_ops, import_taichi
from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import Array
from brainpy._src.math.op_register import (compile_cpu_signature_with_numba,
                                           register_general_batching,
                                           XLACustomOp)
from brainpy._src.math.sparse._utils import csr_to_coo
from brainpy.errors import GPUOperatorNotFound

ti = import_taichi()

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
    """Product of CSR sparse matrix and a dense matrix.

    Args:
        data : array of shape ``(nse,)``.
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
    return raw_csrmm_taichi(data, indices, indptr, matrix, shape=shape, transpose=transpose)[0]


def raw_csrmm_taichi(
        data: Union[float, jnp.ndarray, Array],
        indices: Union[jnp.ndarray, Array],
        indptr: Union[jnp.ndarray, Array],
        matrix: Union[jnp.ndarray, Array],
        *,
        shape: Tuple[int, int],
        transpose: bool = False,
):
    assert len(shape) == 2

    indices = as_jax(indices)
    indptr = as_jax(indptr)
    matrix = as_jax(matrix)

    if matrix.dtype == jnp.bool_:
        matrix = as_jax(matrix, dtype=data.dtype)

    if data.dtype != matrix.dtype:
        raise TypeError('The types of data and vector should be the same. '
                        f'But we got {data.dtype} != {matrix.dtype}.')
    assert data.ndim == indices.ndim == indptr.ndim == 1
    assert matrix.ndim == 2
    assert data.shape == indices.shape
    assert indptr.shape[0] == shape[0] + 1
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise ValueError('indices should be a 1D vector with integer type.')
    if not jnp.issubdtype(indptr.dtype, jnp.integer):
        raise ValueError('indptr should be a 1D vector with integer type.')

    out_shape = shape[1] if transpose else shape[0]
    result_shape = (out_shape, matrix.shape[1])
    # if the shape of indices is (0,), then we return a zero matrix
    if indices.shape[0] == 0:
        return [jnp.zeros(result_shape, dtype=data.dtype),]

    assert matrix.shape[0] == (shape[0] if transpose else shape[1])
    if transpose: 
        prim = _csr_matmat_transpose_heter_p
    else:
        prim = _csr_matmat_heter_p
    return prim(data,
                indices,
                indptr,
                matrix,
                outs=[jax.ShapeDtypeStruct(result_shape, dtype=data.dtype)],
                transpose=transpose,
                shape=shape)


# CPU kernels

@ti.kernel
def _csr_matmat_transpose_heter_cpu(values: ti.types.ndarray(ndim=1),
                                    col_indices: ti.types.ndarray(ndim=1),
                                    row_ptr: ti.types.ndarray(ndim=1),
                                    matrix: ti.types.ndarray(ndim=2),
                                    out: ti.types.ndarray(ndim=2)):
    for col_i in range(out.shape[1]):
        for row_k in range(out.shape[0]):
            r = 0.
            for row_j in range(matrix.shape[0]):
                val = 0.
                for j in range(row_ptr[row_j], row_ptr[row_j + 1]):
                    if col_indices[j] == row_k:
                        val = values[j]
                r += val * matrix[row_j, col_i]
            out[row_k, col_i] = r


@ti.kernel
def _csr_matmat_heter_cpu(values: ti.types.ndarray(ndim=1),
                          col_indices: ti.types.ndarray(ndim=1),
                          row_ptr: ti.types.ndarray(ndim=1),
                          matrix: ti.types.ndarray(ndim=2),
                          out: ti.types.ndarray(ndim=2)):
    for row_i in range(row_ptr.shape[0] - 1):
        for col_k in range(matrix.shape[1]):
            r = 0.
            for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
                r += values[j] * matrix[col_indices[j], col_k]
            out[row_i, col_k] = r


# GPU kernels

@ti.kernel
def _csr_matmat_transpose_heter_gpu(values: ti.types.ndarray(ndim=1),
                                    col_indices: ti.types.ndarray(ndim=1),
                                    row_ptr: ti.types.ndarray(ndim=1),
                                    matrix: ti.types.ndarray(ndim=2),
                                    out: ti.types.ndarray(ndim=2)):
    for row_i in range(row_ptr.shape[0] - 1):
        for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
            for k in range(matrix.shape[0]):
                out[k, col_indices[j]] += values[j] * matrix[k, row_i]


@ti.kernel
def _csr_matmat_heter_gpu(values: ti.types.ndarray(ndim=1),
                          col_indices: ti.types.ndarray(ndim=1),
                          row_ptr: ti.types.ndarray(ndim=1),
                          matrix: ti.types.ndarray(ndim=2),
                          out: ti.types.ndarray(ndim=2)):
    for row_i in range(row_ptr.shape[0] - 1):
        for col_k in range(matrix.shape[1]):
            r = 0.
            for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
                r += values[j] * matrix[col_indices[j], col_k]
            out[row_i, col_k] = r


def _csr_matmat_jvp_values(val_dot, values, col_indices, row_ptr, matrix, *, outs, transpose, shape):
    return raw_csrmm_taichi(val_dot, col_indices, row_ptr, matrix, shape=shape, transpose=transpose)


def _csr_matmat_jvp_matrix(mat_dot, values, col_indices, row_ptr, matrix, *, outs, transpose, shape):
    return raw_csrmm_taichi(values, col_indices, row_ptr, mat_dot, shape=shape, transpose=transpose)


def _csr_matmat_transpose(
        ct, data, indices, indptr, matrix, *, outs, transpose, shape,
):
    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(matrix):
        ct_matrix = raw_csrmm_taichi(data, indices, indptr, ct[0], shape=shape, transpose=not transpose)[0]
        return data, indices, indptr, ct_matrix

    else:
        matrix = jnp.asarray(matrix)
        row, col = csr_to_coo(indices, indptr)
        return (ct[0][row] * matrix[col]).sum(1), indices, indptr, matrix


def _define_op(cpu_kernel, gpu_kernel):
    prim = XLACustomOp(cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel)
    prim.defjvp(_csr_matmat_jvp_values, None, None, _csr_matmat_jvp_matrix)
    prim.def_transpose_rule(_csr_matmat_transpose)
    return prim


# transpose heter
_csr_matmat_transpose_heter_p = _define_op(cpu_kernel=_csr_matmat_transpose_heter_cpu,
                                           gpu_kernel=_csr_matmat_transpose_heter_gpu)

# no transpose heter
_csr_matmat_heter_p = _define_op(cpu_kernel=_csr_matmat_heter_cpu,
                                 gpu_kernel=_csr_matmat_heter_gpu)
