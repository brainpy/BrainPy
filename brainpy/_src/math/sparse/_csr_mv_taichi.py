# -*- coding: utf-8 -*-


from typing import Union, Tuple

import jax
from jax import numpy as jnp
from jax.interpreters import ad

from brainpy._src.dependency_check import import_taichi
from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import Array
from brainpy._src.math.op_register import XLACustomOp
from brainpy._src.math.sparse._utils import csr_to_coo

ti = import_taichi()

__all__ = [
  'csrmv_taichi',
]


# -------------
# CPU operators
# -------------


@ti.kernel
def _sparse_csr_matvec_transpose_homo_cpu(values: ti.types.ndarray(ndim=1),
                                          col_indices: ti.types.ndarray(ndim=1),
                                          row_ptr: ti.types.ndarray(ndim=1),
                                          vector: ti.types.ndarray(ndim=1),
                                          out: ti.types.ndarray(ndim=1)):
  value = values[0]
  ti.loop_config(serialize=True)
  for row_i in range(row_ptr.shape[0] - 1):
    for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
      out[col_indices[j]] += value * vector[row_i]


@ti.kernel
def _sparse_csr_matvec_transpose_heter_cpu(values: ti.types.ndarray(ndim=1),
                                           col_indices: ti.types.ndarray(ndim=1),
                                           row_ptr: ti.types.ndarray(ndim=1),
                                           vector: ti.types.ndarray(ndim=1),
                                           out: ti.types.ndarray(ndim=1)):
  ti.loop_config(serialize=True)
  for row_i in range(row_ptr.shape[0] - 1):
    for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
      out[col_indices[j]] += vector[row_i] * values[j]


@ti.kernel
def _sparse_csr_matvec_homo_cpu(values: ti.types.ndarray(ndim=1),
                                col_indices: ti.types.ndarray(ndim=1),
                                row_ptr: ti.types.ndarray(ndim=1),
                                vector: ti.types.ndarray(ndim=1),
                                out: ti.types.ndarray(ndim=1)):
  value = values[0]
  # ti.loop_config(serialize=True)
  for row_i in range(row_ptr.shape[0] - 1):
    r = 0.
    for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
      r += vector[col_indices[j]]
    out[row_i] = r * value


@ti.kernel
def _sparse_csr_matvec_heter_cpu(values: ti.types.ndarray(ndim=1),
                                 col_indices: ti.types.ndarray(ndim=1),
                                 row_ptr: ti.types.ndarray(ndim=1),
                                 vector: ti.types.ndarray(ndim=1),
                                 out: ti.types.ndarray(ndim=1)):
  # ti.loop_config(serialize=True)
  for row_i in range(row_ptr.shape[0] - 1):
    r = 0.
    for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
      r += values[j] * vector[col_indices[j]]
    out[row_i] = r


# -------------
# GPU operators
# -------------


@ti.kernel
def _sparse_csr_matvec_transpose_homo_gpu(values: ti.types.ndarray(ndim=1),
                                          col_indices: ti.types.ndarray(ndim=1),
                                          row_ptr: ti.types.ndarray(ndim=1),
                                          vector: ti.types.ndarray(ndim=1),
                                          out: ti.types.ndarray(ndim=1)):
  value = values[0]
  for i in range((row_ptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    j = row_ptr[row_i] + index
    end_index = row_ptr[row_i + 1]
    while j < end_index:
      out[col_indices[j]] += value * vector[row_i]
      j += 32


@ti.kernel
def _sparse_csr_matvec_homo_gpu(values: ti.types.ndarray(ndim=1),
                                col_indices: ti.types.ndarray(ndim=1),
                                row_ptr: ti.types.ndarray(ndim=1),
                                vector: ti.types.ndarray(ndim=1),
                                out: ti.types.ndarray(ndim=1)):
  value = values[0]
  for i in range((row_ptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    r = 0.
    j = row_ptr[row_i] + index
    end_index = row_ptr[row_i + 1]
    while j < end_index:
      r += vector[col_indices[j]]
      j += 32
    out[row_i] += value * r


@ti.kernel
def _sparse_csr_matvec_transpose_heter_gpu(values: ti.types.ndarray(ndim=1),
                                           col_indices: ti.types.ndarray(ndim=1),
                                           row_ptr: ti.types.ndarray(ndim=1),
                                           vector: ti.types.ndarray(ndim=1),
                                           out: ti.types.ndarray(ndim=1)):
  for i in range((row_ptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    j = row_ptr[row_i] + index
    end_index = row_ptr[row_i + 1]
    while j < end_index:
      out[col_indices[j]] += values[j] * vector[row_i]
      j += 32


@ti.kernel
def _sparse_csr_matvec_heter_gpu(values: ti.types.ndarray(ndim=1),
                                 col_indices: ti.types.ndarray(ndim=1),
                                 row_ptr: ti.types.ndarray(ndim=1),
                                 vector: ti.types.ndarray(ndim=1),
                                 out: ti.types.ndarray(ndim=1)):
  for i in range((row_ptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    r = 0.
    j = row_ptr[row_i] + index
    end_index = row_ptr[row_i + 1]
    while j < end_index:
      r += values[j] * vector[col_indices[j]]
      j += 32
    out[row_i] += r  # TODO: warp-level primitive


def _sparse_csr_matvec_jvp_values(val_dot, values, col_indices, row_ptr, vector, *, outs, transpose, shape):
  return csrmv_taichi(val_dot, col_indices, row_ptr, vector, shape=shape, transpose=transpose)


def _sparse_csr_matvec_jvp_vector(vec_dot, values, col_indices, row_ptr, vector, *, outs, transpose, shape):
  return csrmv_taichi(values, col_indices, row_ptr, vec_dot, shape=shape, transpose=transpose)


def _sparse_csr_matvec_transpose(
    ct, data, indices, indptr, vector, *, outs, transpose, shape,
):
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices.")
  if ad.is_undefined_primal(vector):
    ct_vector = csrmv_taichi(data, indices, indptr, ct[0], shape=shape, transpose=not transpose)[0]
    return data, indices, indptr, (ad.Zero(vector) if type(ct[0]) is ad.Zero else ct_vector)

  else:
    if type(ct[0]) is ad.Zero:
      ct_data = ad.Zero(data)
    else:
      if data.aval.shape[0] == 1:  # scalar
        ct_data = csrmv_taichi(jnp.ones(1), indices, indptr, vector, shape=shape, transpose=transpose)[0]
        ct_data = jnp.inner(ct[0], ct_data)
      else:
        row, col = csr_to_coo(indices, indptr)
        ct_data = vector[row] * ct[0][col] if transpose else vector[col] * ct[0][row]

    return ct_data, indices, indptr, vector


def csrmv_taichi(
    data: Union[float, jnp.ndarray, Array],
    indices: Union[jnp.ndarray, Array],
    indptr: Union[jnp.ndarray, Array],
    vector: Union[jnp.ndarray, Array],
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
) -> jax.Array:
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

  if transpose:
    if data.shape[0] == 1:
      prim = _csr_matvec_transpose_homo_p
    else:
      prim = _csr_matvec_transpose_heter_p
  else:
    if data.shape[0] == 1:
      prim = _csr_matvec_homo_p
    else:
      prim = _csr_matvec_heter_p

  return prim(data,
              indices,
              indptr,
              vector,
              outs=[jax.ShapeDtypeStruct((out_shape,), dtype=data.dtype)],
              transpose=transpose,
              shape=shape)


def _define_op(cpu_kernel, gpu_kernel):
  prim = XLACustomOp(cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel)
  prim.defjvp(_sparse_csr_matvec_jvp_values, None, None, _sparse_csr_matvec_jvp_vector)
  prim.def_transpose_rule(_sparse_csr_matvec_transpose)
  return prim


# transpose homo
_csr_matvec_transpose_homo_p = _define_op(cpu_kernel=_sparse_csr_matvec_transpose_homo_cpu,
                                          gpu_kernel=_sparse_csr_matvec_transpose_homo_gpu)

# no transpose homo
_csr_matvec_homo_p = _define_op(cpu_kernel=_sparse_csr_matvec_homo_cpu,
                                gpu_kernel=_sparse_csr_matvec_homo_gpu)

# transpose heter
_csr_matvec_transpose_heter_p = _define_op(cpu_kernel=_sparse_csr_matvec_transpose_heter_cpu,
                                           gpu_kernel=_sparse_csr_matvec_transpose_heter_gpu)

# no transpose heter
_csr_matvec_heter_p = _define_op(cpu_kernel=_sparse_csr_matvec_heter_cpu,
                                 gpu_kernel=_sparse_csr_matvec_heter_gpu)