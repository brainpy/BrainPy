# -*- coding: utf-8 -*-


import warnings
from functools import partial
from typing import Union, Tuple

import numba
import numpy as np
from jax import core, numpy as jnp, dtypes, default_backend
from jax.interpreters import ad, mlir, xla
from jax.lib import xla_client
from jaxlib import gpu_sparse

from brainpylib._src.op_register import (compile_cpu_signature_with_numba,
                                         register_general_batching)
from brainpylib._src.sparse_ops.utils import csr_to_coo
from brainpylib._src.tools import transform_brainpy_array

__all__ = [
  'cusparse_csr_matvec',
  'cusparse_coo_matvec',
]


def cusparse_csr_matvec(
    data: Union[float, jnp.ndarray],
    indices: jnp.ndarray,
    indptr: jnp.ndarray,
    vector: jnp.ndarray,
    *,
    shape: Tuple[int, int],
    transpose: bool = False
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

  data = transform_brainpy_array(data)
  indices = transform_brainpy_array(indices)
  indptr = transform_brainpy_array(indptr)
  vector = transform_brainpy_array(vector)
  # checking
  data = jnp.atleast_1d(data)
  if len(shape) != 2:
    raise ValueError(f'shape should be a tuple of int denoting (n_row, n_col). Got {shape}.')
  if not (vector.ndim == data.ndim == indices.ndim == indptr.ndim == 1):
    raise ValueError('Data dimension mismatch. All must be 1D array.')
  if data.shape[0] not in [1, indices.shape[0]]:
    raise ValueError('The size of values should be 1 or be consistent with indices.'
                     f'But we got {data.shape} != {indices.shape}, {data.shape} != 1.')
  # TODO: Change subtype of integer into int32 & uint32
  if not jnp.issubdtype(indices.dtype, jnp.integer):
    raise ValueError('indices should be a 1D vector with integer type.')
  if not jnp.issubdtype(indptr.dtype, jnp.integer):
    raise ValueError('indptr should be a 1D vector with integer type.')
  if default_backend() != 'cpu':
    if data.shape[0] == 1:
      data = jnp.ones(indices.shape, dtype=data.dtype) * data
    if indices.dtype in [jnp.uint32, jnp.uint64]:
      indices = jnp.asarray(indices, dtype=dtypes.canonicalize_dtype(jnp.int64))
    if indptr.dtype in [jnp.uint32, jnp.uint64]:
      indptr = jnp.asarray(indptr, dtype=dtypes.canonicalize_dtype(jnp.int64))
  if data.dtype != vector.dtype:
    raise ValueError(f'Types of data and vector mismatch. Got {data.dtype} != {vector.dtype}.')
  if indptr.shape[0] != shape[0] + 1:
    raise ValueError(f'shape {shape} does not match the given indptr {indptr.shape}.')
  if vector.shape[0] != (shape[0] if transpose else shape[1]):
    raise ValueError(f'shape {shape} does not match the given vector {vector.shape}.')
  # computing
  return csr_matvec_p.bind(data, indices, indptr, vector, shape=shape, transpose=transpose)


def cusparse_coo_matvec(
    data: Union[float, jnp.ndarray],
    row: jnp.ndarray,
    col: jnp.ndarray,
    vector: jnp.ndarray,
    *,
    shape: Tuple[int, int],
    rows_sorted: bool = False,
    cols_sorted: bool = False,
    transpose: bool = False
):
  """Product of COO sparse matrix and a dense vector using cuSPARSE algorithm.

  This function supports JAX transformations, including `jit()`, `grad()`,
  `vmap()` and `pmap()`.

  Parameters
  ----------
  data: ndarray, float
    An array of shape ``(nse,)``.
  row: ndarray
    An array of shape ``(nse,)``.
  col: ndarray
    An array of shape ``(nse,)`` and dtype ``row.dtype``.
  vector: ndarray
    An array of shape ``(shape[0] if transpose else shape[1],)`` and
    dtype ``data.dtype``.
  shape: tuple of int
    The shape of the sparse matrix.
  rows_sorted: bool
    Row index are sorted.
  cols_sorted: bool
    Column index are sorted.
  transpose: bool
    A boolean specifying whether to transpose the sparse matrix
    before computing.

  Returns
  -------
  y: ndarray
    An array of shape ``(shape[1] if transpose else shape[0],)`` representing
    the matrix vector product.
  """
  data = transform_brainpy_array(data)
  row = transform_brainpy_array(row)
  col = transform_brainpy_array(col)
  vector = transform_brainpy_array(vector)
  # checking
  data = jnp.atleast_1d(data)
  if len(shape) != 2:
    raise ValueError(f'shape should be a tuple of int denoting (n_row, n_col). Got {shape}.')
  if not (vector.ndim == data.ndim == row.ndim == col.ndim == 1):
    raise ValueError('Data dimension mismatch. All must be 1D array.')
  if data.shape[0] not in [1, row.shape[0]]:
    raise ValueError('The size of values should be 1 or be consistent with indices.'
                     f'But we got {data.shape} != {row.shape}, {data.shape} != 1.')
  if row.shape != col.shape:
    raise ValueError(f'The size of row and col mismatch. {row.shape} != {col.shape}.')
  # TODO: Change subtype of integer into int32 & uint32
  if not jnp.issubdtype(row.dtype, jnp.integer):
    raise ValueError('row should be a 1D vector with integer type.')
  if not jnp.issubdtype(col.dtype, jnp.integer):
    raise ValueError('col should be a 1D vector with integer type.')
  if default_backend() != 'cpu':
    if data.shape[0] == 1:
      data = jnp.ones(row.shape, dtype=data.dtype) * data
    if row.dtype in [jnp.uint32, jnp.uint64]:
      row = jnp.asarray(row, dtype=dtypes.canonicalize_dtype(jnp.int64))
    if col.dtype in [jnp.uint32, jnp.uint64]:
      col = jnp.asarray(col, dtype=dtypes.canonicalize_dtype(jnp.int64))
  if data.dtype != vector.dtype:
    raise ValueError(f'Types of data and vector mismatch. Got {data.dtype} != {vector.dtype}.')
  if vector.shape[0] != (shape[0] if transpose else shape[1]):
    raise ValueError(f'shape {shape} does not match the given vector {vector.shape}.')

  # computing
  return coo_matvec_p.bind(data,
                           row,
                           col,
                           vector,
                           shape=shape,
                           rows_sorted=rows_sorted,
                           cols_sorted=cols_sorted,
                           transpose=transpose)


# --------------------------------------------------------------------
# cusparse_csr_matvec
# --------------------------------------------------------------------

# operator for `cusparse_csr_matvec` #
def _csr_matvec_numba_abstract(data, indices, indptr, v, *, shape, transpose):
  out_shape = shape[1] if transpose else shape[0]
  return core.ShapedArray((out_shape,), data.dtype)


@numba.njit(fastmath=True)
def _csr_matvec_transpose_numba_imp(outs, ins):
  res_val = outs
  res_val.fill(0)
  values, col_indices, row_ptr, vector, shape, _ = ins
  # (csr mat).T @ vec

  if values.shape[0] == 1:
    values = values[0]
    for row_i in range(shape[0]):
      v = vector[row_i]
      for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
        res_val[col_indices[j]] += values * v
  else:
    for row_i in range(shape[0]):
      v = vector[row_i]
      for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
        res_val[col_indices[j]] += v * values[j]


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _csr_matvec_numba_imp(outs, ins):
  res_val = outs
  res_val.fill(0)
  values, col_indices, row_ptr, vector, shape, _ = ins
  # csr mat @ vec
  if values.shape[0] == 1:
    values = values[0]
    for row_i in numba.prange(shape[0]):
      r = 0.
      for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
        r += values * vector[col_indices[j]]
      res_val[row_i] = r
  else:
    for row_i in numba.prange(shape[0]):
      r = 0.
      for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
        r += values[j] * vector[col_indices[j]]
      res_val[row_i] = r


def _csr_matvec_cpu_translation(c, data, indices, indptr, vector, *, shape, transpose):
  inputs = (data, indices, indptr, vector)
  description = dict(shape=shape, transpose=transpose)
  if transpose:
    target_name, inputs, input_layouts, output_layouts = compile_cpu_signature_with_numba(
      c,
      _csr_matvec_transpose_numba_imp,
      _csr_matvec_numba_abstract,
      multiple_results=False,
      inputs=inputs,
      description=description
    )
  else:
    target_name, inputs, input_layouts, output_layouts = compile_cpu_signature_with_numba(
      c,
      _csr_matvec_numba_imp,
      _csr_matvec_numba_abstract,
      multiple_results=False,
      inputs=inputs,
      description=description
    )
  return xla_client.ops.CustomCallWithLayout(
    c,
    target_name,
    operands=inputs,
    operand_shapes_with_layout=input_layouts,
    shape_with_layout=output_layouts,
  )


def _csr_matvec_gpu_lowering(
    ctx, data, indices, indptr, v,
    *, shape, transpose
):
  data_aval, indices_aval, _, v_aval = ctx.avals_in
  dtype = data_aval.dtype
  if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
    raise TypeError(f"cusparse_csr_matvec cusparse/hipsparse lowering not available for dtype={dtype}. "
                    "Falling back to default implementation.")
  return [gpu_sparse.cuda_csr_matvec(data, indices, indptr, v,
                                     shape=shape, transpose=transpose,
                                     data_dtype=dtype, x_dtype=v_aval.dtype,
                                     index_dtype=indices_aval.dtype)]


def _csr_matvec_jvp_mat(data_dot, data, indices, indptr, v, *, shape, transpose):
  return cusparse_csr_matvec(data_dot, indices, indptr, v, shape=shape, transpose=transpose)


def _csr_matvec_jvp_vec(v_dot, data, indices, indptr, v, *, shape, transpose):
  return cusparse_csr_matvec(data, indices, indptr, v_dot, shape=shape, transpose=transpose)


def _csr_matvec_transpose(ct, data, indices, indptr, vector, *, shape, transpose):
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices.")

  if ad.is_undefined_primal(vector):
    ct_vector = cusparse_csr_matvec(data, indices, indptr, ct, shape=shape, transpose=not transpose)
    return data, indices, indptr, (ad.Zero(vector) if type(ct) is ad.Zero else ct_vector)

  else:
    if type(ct) is ad.Zero:
      ct_data = ad.Zero(data)
    else:
      if data.aval.shape[0] == 1:  # scalar
        ct_data = cusparse_csr_matvec(jnp.ones(1), indices, indptr, vector, shape=shape, transpose=transpose)
        ct_data = jnp.inner(ct, ct_data)
      else:  # heterogeneous values
        row, col = csr_to_coo(indices, indptr)
        ct_data = vector[row] * ct[col] if transpose else vector[col] * ct[row]
    return ct_data, indices, indptr, vector


csr_matvec_p = core.Primitive('cusparse_csr_matvec')
csr_matvec_p.def_abstract_eval(_csr_matvec_numba_abstract)
csr_matvec_p.def_impl(partial(xla.apply_primitive, csr_matvec_p))
xla.backend_specific_translations['cpu'][csr_matvec_p] = _csr_matvec_cpu_translation
ad.defjvp(csr_matvec_p, _csr_matvec_jvp_mat, None, None, _csr_matvec_jvp_vec)
ad.primitive_transposes[csr_matvec_p] = _csr_matvec_transpose
register_general_batching(csr_matvec_p)
if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(csr_matvec_p, _csr_matvec_gpu_lowering, platform='cuda')


# --------------------------------------------------------------------
# cusparse_coo_matvec


def _coo_matvec_impl(data, row, col, v, *, shape, rows_sorted, cols_sorted, transpose):
  v = jnp.asarray(v)
  if transpose:
    row, col = col, row
  out_shape = shape[1] if transpose else shape[0]
  dv = data * v[col]
  return jnp.zeros(out_shape, dv.dtype).at[row].add(dv)


def _coo_matvec_abstract_eval(data, row, col, v, *, shape, rows_sorted, cols_sorted, transpose):
  assert data.shape == row.shape == col.shape
  assert data.dtype == v.dtype
  assert row.dtype == col.dtype
  assert len(shape) == 2
  assert v.ndim == 1
  assert v.shape[0] == (shape[0] if transpose else shape[1])
  out_shape = shape[1] if transpose else shape[0]
  return core.ShapedArray((out_shape,), data.dtype)


_coo_matvec_lowering = mlir.lower_fun(_coo_matvec_impl, multiple_results=False)


def _coo_matvec_gpu_lowering(coo_matvec_mhlo, ctx, data, row, col, v, *,
                             shape, rows_sorted, cols_sorted, transpose):
  data_aval, row_aval, _, x_aval = ctx.avals_in
  dtype = data_aval.dtype
  if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
    warnings.warn(f"cusparse_coo_matvec cusparse/hipsparse lowering not available for dtype={dtype}. "
                  "Falling back to default implementation.", UserWarning)
    return _coo_matvec_lowering(ctx, data, row, col, v,
                                shape=shape,
                                rows_sorted=rows_sorted,
                                cols_sorted=cols_sorted,
                                transpose=transpose)

  if rows_sorted:
    shape = shape
  elif cols_sorted:
    row, col = col, row
    transpose = not transpose
    shape = shape[::-1]
  else:
    warnings.warn("cusparse_coo_matvec GPU lowering requires matrices with sorted rows or sorted cols. "
                  "To sort the rows in your matrix, use e.g. mat = mat._sort_rows(). Falling "
                  "back to the default implementation.", UserWarning)
    return _coo_matvec_lowering(ctx, data, row, col, v,
                                shape=shape,
                                rows_sorted=rows_sorted,
                                cols_sorted=cols_sorted,
                                transpose=transpose)

  return [coo_matvec_mhlo(data, row, col, v,
                          shape=shape,
                          transpose=transpose,
                          index_dtype=row_aval.dtype,
                          data_dtype=dtype,
                          x_dtype=x_aval.dtype)]


def _coo_matvec_jvp_mat(data_dot, data, row, col, v, *, shape, rows_sorted, cols_sorted, transpose):
  return cusparse_coo_matvec(data_dot, row, col, v,
                             shape=shape,
                             rows_sorted=rows_sorted,
                             cols_sorted=cols_sorted,
                             transpose=transpose)


def _coo_matvec_jvp_vec(v_dot, data, row, col, v, *, shape, rows_sorted, cols_sorted, transpose):
  return cusparse_coo_matvec(data, row, col, v_dot,
                             shape=shape,
                             rows_sorted=rows_sorted,
                             cols_sorted=cols_sorted,
                             transpose=transpose)


def _coo_matvec_transpose(ct, data, row, col, v, *, shape, rows_sorted, cols_sorted, transpose):
  assert not ad.is_undefined_primal(row)
  assert not ad.is_undefined_primal(col)

  if ad.is_undefined_primal(v):
    return data, row, col, cusparse_coo_matvec(data, row, col, ct,
                                               shape=shape,
                                               rows_sorted=rows_sorted,
                                               cols_sorted=cols_sorted,
                                               transpose=not transpose)
  else:
    return ct[row] * v[col], row, col, v


coo_matvec_p = core.Primitive('cusparse_coo_matvec')
coo_matvec_p.def_abstract_eval(_coo_matvec_abstract_eval)
coo_matvec_p.def_impl(_coo_matvec_impl)
ad.defjvp(coo_matvec_p, _coo_matvec_jvp_mat, None, None, _coo_matvec_jvp_vec)
ad.primitive_transposes[coo_matvec_p] = _coo_matvec_transpose
mlir.register_lowering(coo_matvec_p, _coo_matvec_lowering)
register_general_batching(coo_matvec_p)
if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(
    coo_matvec_p,
    partial(_coo_matvec_gpu_lowering, gpu_sparse.cuda_coo_matvec),
    platform='cuda'
  )
