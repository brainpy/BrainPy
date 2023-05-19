# -*- coding: utf-8 -*-


import warnings
from functools import partial
from typing import Union, Tuple

import numpy as np
from jax import core, numpy as jnp, dtypes, default_backend
from jax.interpreters import ad, mlir
from jaxlib import gpu_sparse

from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import Array
from brainpy._src.math.op_registers import register_general_batching

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
    transpose: bool = False,
    method: str = 'cusparse'
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
  method: str
    The method used to compute the matrix-vector multiplication.

  Returns
  -------
  y: ndarray
    An array of shape ``(shape[1] if transpose else shape[0],)`` representing
    the matrix vector product.
  """

  data = jnp.atleast_1d(as_jax(data))
  row = as_jax(row)
  col = as_jax(col)
  vector = as_jax(vector)

  if method == 'cusparse':
    if default_backend() != 'cpu':
      if data.shape[0] == 1:
        data = jnp.ones(row.shape, dtype=data.dtype) * data
      if row.dtype in [jnp.uint32, jnp.uint64]:
        row = jnp.asarray(row, dtype=dtypes.canonicalize_dtype(jnp.int64))
      if col.dtype in [jnp.uint32, jnp.uint64]:
        col = jnp.asarray(col, dtype=dtypes.canonicalize_dtype(jnp.int64))
    return _coomv_cusparse_p.bind(data,
                                  row,
                                  col,
                                  vector,
                                  shape=shape,
                                  rows_sorted=rows_sorted,
                                  cols_sorted=cols_sorted,
                                  transpose=transpose)

  else:
    raise ValueError


# --------------------------------------------------------------------
# cusparse_coo_matvec


def _coomv_impl(data, row, col, v, *, shape, rows_sorted, cols_sorted, transpose):
  v = jnp.asarray(v)
  if transpose:
    row, col = col, row
  out_shape = shape[1] if transpose else shape[0]
  dv = data * v[col]
  return jnp.zeros(out_shape, dv.dtype).at[row].add(dv)


def _coomv_abstract_eval(data, row, col, v, *, shape, rows_sorted, cols_sorted, transpose):
  assert data.shape == row.shape == col.shape
  assert data.dtype == v.dtype
  assert row.dtype == col.dtype
  assert len(shape) == 2
  assert v.ndim == 1
  assert v.shape[0] == (shape[0] if transpose else shape[1])
  out_shape = shape[1] if transpose else shape[0]
  return core.ShapedArray((out_shape,), data.dtype)


_coo_matvec_lowering = mlir.lower_fun(_coomv_impl, multiple_results=False)


def _coomv_gpu_lowering(coo_matvec_mhlo, ctx, data, row, col, v, *,
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


def _coomv_jvp_mat(data_dot, data, row, col, v, *, shape, rows_sorted, cols_sorted, transpose):
  return _coomv_cusparse_p.bind(data_dot, row, col, v,
                                shape=shape,
                                rows_sorted=rows_sorted,
                                cols_sorted=cols_sorted,
                                transpose=transpose)


def _coomv_jvp_vec(v_dot, data, row, col, v, *, shape, rows_sorted, cols_sorted, transpose):
  return _coomv_cusparse_p.bind(data, row, col, v_dot,
                                shape=shape,
                                rows_sorted=rows_sorted,
                                cols_sorted=cols_sorted,
                                transpose=transpose)


def _coomv_transpose(ct, data, row, col, v, *, shape, rows_sorted, cols_sorted, transpose):
  assert not ad.is_undefined_primal(row)
  assert not ad.is_undefined_primal(col)

  if ad.is_undefined_primal(v):
    return data, row, col, _coomv_cusparse_p.bind(data, row, col, ct,
                                                  shape=shape,
                                                  rows_sorted=rows_sorted,
                                                  cols_sorted=cols_sorted,
                                                  transpose=not transpose)
  else:
    return ct[row] * v[col], row, col, v


_coomv_cusparse_p = core.Primitive('cusparse_coo_matvec')
_coomv_cusparse_p.def_abstract_eval(_coomv_abstract_eval)
_coomv_cusparse_p.def_impl(_coomv_impl)
ad.defjvp(_coomv_cusparse_p, _coomv_jvp_mat, None, None, _coomv_jvp_vec)
ad.primitive_transposes[_coomv_cusparse_p] = _coomv_transpose
mlir.register_lowering(_coomv_cusparse_p, _coo_matvec_lowering)
mlir.register_lowering(_coomv_cusparse_p,
                       partial(_coomv_gpu_lowering, gpu_sparse.cuda_coo_matvec),
                       platform='cuda')
register_general_batching(_coomv_cusparse_p)


