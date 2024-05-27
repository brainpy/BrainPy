# -*- coding: utf-8 -*-


from typing import Union, Tuple

import jax
import numpy as np
from jax import numpy as jnp
from jax.experimental.sparse import csr
from jax.interpreters import ad

from brainpy._src.dependency_check import import_taichi
from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import Array
from brainpy._src.math.op_register import (XLACustomOp, register_general_batching)
from brainpy.errors import PackageMissingError

ti = import_taichi(error_if_not_found=False)

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
  """
  Product of CSR sparse matrix and a dense matrix.

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
      representing the matrix-matrix product.
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
  data = jnp.atleast_1d(data)

  if matrix.dtype == jnp.bool_:
    matrix = as_jax(matrix, dtype=data.dtype)

  if data.dtype != matrix.dtype:
    raise TypeError('The types of data and vector should be the same. '
                    f'But we got {data.dtype} != {matrix.dtype}.')
  assert matrix.ndim == 2

  if np.ndim(data) == 1:
    if data.shape[0] not in [1, indices.shape[0]]:
      raise ValueError('The size of data should be 1 or be consistent with indices.'
                       f'But we got {data.shape} != {indices.shape}, {data.shape} != 1.')
  assert indptr.shape[0] == shape[0] + 1
  if not jnp.issubdtype(indices.dtype, jnp.integer):
    raise ValueError('indices should be a 1D vector with integer type.')
  if not jnp.issubdtype(indptr.dtype, jnp.integer):
    raise ValueError('indptr should be a 1D vector with integer type.')

  out_shape = shape[1] if transpose else shape[0]
  result_shape = (out_shape, matrix.shape[1])

  assert matrix.shape[0] == (shape[0] if transpose else shape[1])

  if indices.shape[0] == 0:
    return [jnp.zeros(result_shape, dtype=data.dtype), ]

  # homo -> taichi,
  # heter -> cusparse
  if data.shape[0] != 1:
    return [_csr_matmat_cusparse_p.bind(data, indices, indptr, matrix, shape=shape, transpose=transpose), ]
  else:
    if ti is None:
      raise PackageMissingError.by_purpose('taichi', 'customzied sparse matrix multiplication')
    if transpose:
      prim = _csr_matmat_transpose_homo_p
    else:
      prim = _csr_matmat_homo_p
    r = prim(indices,
             indptr,
             matrix,
             outs=[jax.ShapeDtypeStruct(result_shape, dtype=matrix.dtype)],
             transpose=transpose,
             shape=shape)
    return [r[0] * data]


# taichi kernels
if ti is not None:
  # @ti.kernel
  # def _csr_matmat_transpose_heter(values: ti.types.ndarray(ndim=1),
  #                                 col_indices: ti.types.ndarray(ndim=1),
  #                                 row_ptr: ti.types.ndarray(ndim=1),
  #                                 matrix: ti.types.ndarray(ndim=2),
  #                                 out: ti.types.ndarray(ndim=2)):
  #   for row_i in range(row_ptr.shape[0] - 1):
  #     for i in range(row_ptr[row_i], row_ptr[row_i + 1]):
  #       col = col_indices[i]
  #       for j in range(out.shape[1]):
  #         out[col, j] += values[row_i] * matrix[row_i, j]
  #
  # @ti.kernel
  # def _csr_matmat_heter(values: ti.types.ndarray(ndim=1),
  #                       col_indices: ti.types.ndarray(ndim=1),
  #                       row_ptr: ti.types.ndarray(ndim=1),
  #                       matrix: ti.types.ndarray(ndim=2),
  #                       out: ti.types.ndarray(ndim=2)):
  #   for row_i, col_k in ti.ndrange(out.shape[0], out.shape[1]):
  #     r = 0.
  #     for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
  #       r += values[j] * matrix[col_indices[j], col_k]
  #     out[row_i, col_k] = r
  #
  # # transpose heter
  # _csr_matmat_transpose_heter_p = _define_op(cpu_kernel=_csr_matmat_transpose_heter,
  #                                            gpu_kernel=_csr_matmat_transpose_heter)
  #
  # # no transpose heter
  # _csr_matmat_heter_p = _define_op(cpu_kernel=_csr_matmat_heter,
  #                                  gpu_kernel=_csr_matmat_heter)

  @ti.kernel
  def _csr_matmat_transpose_homo_cpu(col_indices: ti.types.ndarray(ndim=1),
                                     row_ptr: ti.types.ndarray(ndim=1),
                                     matrix: ti.types.ndarray(ndim=2),
                                     out: ti.types.ndarray(ndim=2)):
    # matrix: (k, n)
    # sparse matrix: (m, k)
    n = out.shape[1]
    m = row_ptr.shape[0] - 1
    for j in range(n):  # parallize along the n dimension
      for row_i in range(m):  # loop along the m dimension
        for i in range(row_ptr[row_i], row_ptr[row_i + 1]):
          out[col_indices[i], j] += matrix[row_i, j]


  @ti.kernel
  def _csr_matmat_transpose_homo_gpu(col_indices: ti.types.ndarray(ndim=1),
                                     row_ptr: ti.types.ndarray(ndim=1),
                                     matrix: ti.types.ndarray(ndim=2),
                                     out: ti.types.ndarray(ndim=2)):
    m = row_ptr.shape[0] - 1
    n = matrix.shape[1]
    for j, row_i in ti.ndrange(n, m):  # paralleize along the (n and m) dimensions
      for i in range(row_ptr[row_i], row_ptr[row_i + 1]):
        out[col_indices[i], j] += matrix[row_i, j]


  @ti.kernel
  def _csr_matmat_homo(col_indices: ti.types.ndarray(ndim=1),
                       row_ptr: ti.types.ndarray(ndim=1),
                       matrix: ti.types.ndarray(ndim=2),
                       out: ti.types.ndarray(ndim=2)):
    # matrix: (k, n)
    # sparse matrix: (m, k)
    m, n = out.shape
    for row_i, col_k in ti.ndrange(m, n):
      r = 0.
      for row_j in range(row_ptr[row_i], row_ptr[row_i + 1]):
        r += matrix[col_indices[row_j], col_k]
      out[row_i, col_k] = r


  def _csr_matmat_jvp_matrix(mat_dot, col_indices, row_ptr, matrix, *, outs, transpose, shape):
    if transpose:
      return _csr_matmat_transpose_homo_p(col_indices, row_ptr, mat_dot, shape=shape, transpose=transpose, outs=outs)
    else:
      return _csr_matmat_homo_p(col_indices, row_ptr, mat_dot, shape=shape, transpose=transpose, outs=outs)


  def _csr_matmat_transpose(
      ct, col_indices, row_ptr, matrix, *, outs, transpose, shape,
  ):
    if ad.is_undefined_primal(col_indices) or ad.is_undefined_primal(row_ptr):
      raise ValueError("Cannot transpose with respect to sparse indices.")
    assert ad.is_undefined_primal(matrix)
    ct_matrix = raw_csrmm_taichi(jnp.ones(1), col_indices, row_ptr, ct[0],
                                 shape=shape,
                                 transpose=not transpose)
    return col_indices, row_ptr, (ad.Zero(matrix) if type(ct[0]) is ad.Zero else ct_matrix[0])


  def _define_op(cpu_kernel, gpu_kernel):
    prim = XLACustomOp(cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel)
    prim.defjvp(None, None, _csr_matmat_jvp_matrix)
    prim.def_transpose_rule(_csr_matmat_transpose)
    return prim


  # transpose homo
  _csr_matmat_transpose_homo_p = _define_op(cpu_kernel=_csr_matmat_transpose_homo_cpu,
                                            gpu_kernel=_csr_matmat_transpose_homo_gpu)

  # no transpose homo
  _csr_matmat_homo_p = _define_op(cpu_kernel=_csr_matmat_homo, gpu_kernel=_csr_matmat_homo)

  # heter CUSPARSE
  _csr_matmat_cusparse_p = csr.csr_matmat_p
  register_general_batching(_csr_matmat_cusparse_p)
