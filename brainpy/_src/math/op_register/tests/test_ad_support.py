import pytest
from typing import Tuple

import jax
from jax import core
from jax import numpy as jnp
from jax.interpreters import ad

import brainpy as bp
import brainpy.math as bm
from brainpy._src.dependency_check import import_numba

numba = import_numba(error_if_not_found=False)
if numba is None:
  pytest.skip('no numba', allow_module_level=True)

bm.set_platform('cpu')


def csrmv(data, indices, indptr, vector, *, shape: Tuple[int, int], transpose: bool = False, ):
  data = jnp.atleast_1d(bm.as_jax(data))
  indices = bm.as_jax(indices)
  indptr = bm.as_jax(indptr)
  vector = bm.as_jax(vector)
  if vector.dtype == jnp.bool_:
    vector = bm.as_jax(vector, dtype=data.dtype)
  outs = [core.ShapedArray([shape[1] if transpose else shape[0]], data.dtype)]
  if transpose:
    return prim_trans(data, indices, indptr, vector, outs=outs, shape=shape, transpose=transpose)
  else:
    return prim(data, indices, indptr, vector, outs=outs, shape=shape, transpose=transpose)


@numba.njit(fastmath=True)
def _csr_matvec_transpose_numba_imp(values, col_indices, row_ptr, vector, res_val):
  res_val.fill(0)
  if values.shape[0] == 1:
    values = values[0]
    for row_i in range(vector.shape[0]):
      v = vector[row_i]
      for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
        res_val[col_indices[j]] += values * v
  else:
    for row_i in range(vector.shape[0]):
      v = vector[row_i]
      for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
        res_val[col_indices[j]] += v * values[j]


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _csr_matvec_numba_imp(values, col_indices, row_ptr, vector, res_val):
  res_val.fill(0)
  # csr mat @ vec
  if values.shape[0] == 1:
    values = values[0]
    for row_i in numba.prange(res_val.shape[0]):
      r = 0.
      for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
        r += values * vector[col_indices[j]]
      res_val[row_i] = r
  else:
    for row_i in numba.prange(res_val.shape[0]):
      r = 0.
      for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
        r += values[j] * vector[col_indices[j]]
      res_val[row_i] = r


def _csrmv_jvp_mat(data_dot, data, indices, indptr, v, *, shape, transpose, **kwargs):
  return csrmv(data_dot, indices, indptr, v, shape=shape, transpose=transpose)


def _csrmv_jvp_vec(v_dot, data, indices, indptr, v, *, shape, transpose, **kwargs):
  return csrmv(data, indices, indptr, v_dot, shape=shape, transpose=transpose)


def _csrmv_cusparse_transpose(ct, data, indices, indptr, vector, *, shape, transpose, **kwargs):
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices.")

  ct = ct[0]
  if ad.is_undefined_primal(vector):
    ct_vector = csrmv(data, indices, indptr, ct, shape=shape, transpose=not transpose)
    return data, indices, indptr, (ad.Zero(vector) if type(ct) is ad.Zero else ct_vector)

  else:
    if type(ct) is ad.Zero:
      ct_data = ad.Zero(data)
    else:
      if data.aval.shape[0] == 1:  # scalar
        ct_data = csrmv(jnp.ones(1), indices, indptr, vector, shape=shape, transpose=transpose)
        ct_data = jnp.inner(ct, ct_data)
      else:  # heterogeneous values
        row, col = bm.sparse.csr_to_coo(indices, indptr)
        ct_data = vector[row] * ct[col] if transpose else vector[col] * ct[row]
    return ct_data, indices, indptr, vector


prim_trans = bm.XLACustomOp(_csr_matvec_transpose_numba_imp)
prim_trans.defjvp(_csrmv_jvp_mat, None, None, _csrmv_jvp_vec)
prim_trans.def_transpose_rule(_csrmv_cusparse_transpose)

prim = bm.XLACustomOp(_csr_matvec_numba_imp)
prim.defjvp(_csrmv_jvp_mat, None, None, _csrmv_jvp_vec)
prim.def_transpose_rule(_csrmv_cusparse_transpose)


def sum_op(op):
  def func(*args, **kwargs):
    r = op(*args, **kwargs)
    return r.sum()

  return func


def try_a_trial(transpose, shape):
  rng = bm.random.RandomState()
  conn = bp.conn.FixedProb(0.1)
  indices, indptr = conn(*shape).require('pre2post')
  indices = bm.as_jax(indices)
  indptr = bm.as_jax(indptr)
  heter_data = rng.random(indices.shape)
  heter_data = bm.as_jax(heter_data)
  vector = rng.random(shape[0] if transpose else shape[1])
  vector = bm.as_jax(vector)

  r5 = jax.grad(sum_op(lambda *args, **kwargs: bm.sparse.csrmv(*args, **kwargs)), argnums=(0, 3))(
    heter_data, indices, indptr, vector.astype(float), shape=shape, transpose=transpose)
  r6 = jax.grad(sum_op(lambda *args, **kwargs: csrmv(*args, **kwargs)[0]), argnums=(0, 3))(
    heter_data, indices, indptr, vector.astype(float), shape=shape, transpose=transpose)
  print(r5)
  print(r6)
  assert bm.allclose(r5[0], r6[0])
  assert bm.allclose(r5[1], r6[1][0])


def test():
  transposes = [True, False]
  shapes = [(100, 200), (10, 1000), (2, 2000)]

  for transpose in transposes:
    for shape in shapes:
      try_a_trial(transpose, shape)
