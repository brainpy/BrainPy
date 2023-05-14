# -*- coding: utf-8 -*-


import warnings
from functools import partial
from typing import Union, Tuple

import numpy as np
from brainpylib._src.op_register import (register_general_batching)
from jax import core, numpy as jnp, dtypes, default_backend
from jax.interpreters import ad, mlir
from jaxlib import gpu_sparse

from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import Array

__all__ = [
  'coomv',
]

def coomv(
    data: Union[float, jnp.ndarray, Array],
    row: Union[jnp.ndarray, Array],
    col: Union[jnp.ndarray, Array],
    vector: Union[jnp.ndarray, Array],
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
  pass



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
  data = as_jax(data)
  row = as_jax(row)
  col = as_jax(col)
  vector = as_jax(vector)
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
  return cusparse_coo_matvec_p.bind(data,
                                    row,
                                    col,
                                    vector,
                                    shape=shape,
                                    rows_sorted=rows_sorted,
                                    cols_sorted=cols_sorted,
                                    transpose=transpose)


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


cusparse_coo_matvec_p = core.Primitive('cusparse_coo_matvec')
cusparse_coo_matvec_p.def_abstract_eval(_coo_matvec_abstract_eval)
cusparse_coo_matvec_p.def_impl(_coo_matvec_impl)
ad.defjvp(cusparse_coo_matvec_p, _coo_matvec_jvp_mat, None, None, _coo_matvec_jvp_vec)
ad.primitive_transposes[cusparse_coo_matvec_p] = _coo_matvec_transpose
mlir.register_lowering(cusparse_coo_matvec_p, _coo_matvec_lowering)
register_general_batching(cusparse_coo_matvec_p)
if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(
    cusparse_coo_matvec_p,
    partial(_coo_matvec_gpu_lowering, gpu_sparse.cuda_coo_matvec),
    platform='cuda'
  )
