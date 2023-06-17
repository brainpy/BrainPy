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

from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import Array
from brainpy._src.math.op_registers import (compile_cpu_signature_with_numba,
                                            register_general_batching)
from brainpy._src.math.sparse._utils import csr_to_coo
from brainpy.errors import GPUOperatorNotFound

try:
  from brainpylib import gpu_ops
except ImportError:
  gpu_ops = None

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
    method: str = 'cusparse',
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
  method: str
    The method used to compute Matrix-Vector Multiplication. The candidate methods are:

    - ``cusparse``: using cuSPARSE library.
    - ``scalar``:
    - ``vector``:
    - ``adaptive``:

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

  if method == 'cusparse':
    if jax.default_backend() == 'gpu':
      if data.shape[0] == 1:
        data = jnp.ones(indices.shape, dtype=data.dtype) * data
      if indices.dtype in [jnp.uint32, jnp.uint64]:
        indices = jnp.asarray(indices, dtype=dtypes.canonicalize_dtype(jnp.int64))
      if indptr.dtype in [jnp.uint32, jnp.uint64]:
        indptr = jnp.asarray(indptr, dtype=dtypes.canonicalize_dtype(jnp.int64))
    return _csrmv_cusparse_p.bind(data,
                                  indices,
                                  indptr,
                                  vector,
                                  shape=shape,
                                  transpose=transpose)

  elif method == 'adaptive':
    return _csrmv_adaptive_p.bind(data, indices, indptr, vector, shape=shape, transpose=transpose)

  elif method == 'scalar':
    return _csrmv_scalar_p.bind(data, indices, indptr, vector, shape=shape, transpose=transpose)

  elif method == 'vector':
    return _csrmv_vector_p.bind(data, indices, indptr, vector, shape=shape, transpose=transpose)

  else:
    raise ValueError(f'Only support methods: cusparse, scalar, vector, and adaptive. But we got {method}.')


def _csrmv_abstract(data, indices, indptr, vector, *, shape, transpose):
  if data.dtype not in [jnp.float32, jnp.float64]:
    raise TypeError(f'Only support float32 and float64. But we got {data.dtype}.')
  if data.dtype != vector.dtype:
    raise TypeError('The types of data and vector should be the same. '
                    f'But we got {data.dtype} != {vector.dtype}.')
  assert data.ndim == indices.ndim == indptr.ndim == vector.ndim == 1
  if not jnp.issubdtype(indices.dtype, jnp.integer):
    raise ValueError('indices should be a 1D vector with integer type.')
  if not jnp.issubdtype(indptr.dtype, jnp.integer):
    raise ValueError('indptr should be a 1D vector with integer type.')
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


def _csrmv_cpu_translation(c, data, indices, indptr, vector, *, shape, transpose):
  inputs = (data, indices, indptr, vector)
  description = dict(shape=shape, transpose=transpose)
  if transpose:
    target_name, inputs, input_layouts, output_layouts = compile_cpu_signature_with_numba(
      c,
      _csr_matvec_transpose_numba_imp,
      _csrmv_abstract,
      multiple_results=False,
      inputs=inputs,
      description=description
    )
  else:
    target_name, inputs, input_layouts, output_layouts = compile_cpu_signature_with_numba(
      c,
      _csr_matvec_numba_imp,
      _csrmv_abstract,
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


def _csrmv_cusparse_gpu_lowering(ctx, data, indices, indptr, vector, *, shape, transpose):
  data_aval, indices_aval, _, v_aval = ctx.avals_in
  dtype = data_aval.dtype
  if dtype not in [np.float32, np.float64, np.complex64, np.complex128]:
    raise TypeError(f"cusparse_csr_matvec cusparse/hipsparse lowering not available for dtype={dtype}. "
                    "Falling back to default implementation.")
  return [gpu_sparse.cuda_csr_matvec(data, indices, indptr, vector,
                                     shape=shape,
                                     transpose=transpose,
                                     data_dtype=dtype,
                                     x_dtype=v_aval.dtype,
                                     index_dtype=indices_aval.dtype)]


def _csrmv_jvp_mat(csr_prim, data_dot, data, indices, indptr, v, *, shape, transpose):
  return csr_prim.bind(data_dot, indices, indptr, v, shape=shape, transpose=transpose)


def _csrmv_jvp_vec(prim, v_dot, data, indices, indptr, v, *, shape, transpose):
  return prim.bind(data, indices, indptr, v_dot, shape=shape, transpose=transpose)


def _csrmv_cusparse_transpose(ct, data, indices, indptr, vector, *, shape, transpose):
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices.")

  if ad.is_undefined_primal(vector):
    ct_vector = _csrmv_cusparse_p.bind(data, indices, indptr, ct, shape=shape, transpose=not transpose)
    return data, indices, indptr, (ad.Zero(vector) if type(ct) is ad.Zero else ct_vector)

  else:
    if type(ct) is ad.Zero:
      ct_data = ad.Zero(data)
    else:
      if data.aval.shape[0] == 1:  # scalar
        ct_data = _csrmv_cusparse_p.bind(jnp.ones(1), indices, indptr, vector, shape=shape, transpose=transpose)
        ct_data = jnp.inner(ct, ct_data)
      else:  # heterogeneous values
        row, col = csr_to_coo(indices, indptr)
        ct_data = vector[row] * ct[col] if transpose else vector[col] * ct[row]
    return ct_data, indices, indptr, vector


_csrmv_cusparse_p = core.Primitive('cusparse_csr_matvec')
_csrmv_cusparse_p.def_abstract_eval(_csrmv_abstract)
_csrmv_cusparse_p.def_impl(partial(xla.apply_primitive, _csrmv_cusparse_p))
xla.backend_specific_translations['cpu'][_csrmv_cusparse_p] = _csrmv_cpu_translation
ad.defjvp(_csrmv_cusparse_p,
          partial(_csrmv_jvp_mat, _csrmv_cusparse_p),
          None,
          None,
          partial(_csrmv_jvp_vec, _csrmv_cusparse_p))
ad.primitive_transposes[_csrmv_cusparse_p] = _csrmv_cusparse_transpose
register_general_batching(_csrmv_cusparse_p)
mlir.register_lowering(_csrmv_cusparse_p, _csrmv_cusparse_gpu_lowering, platform='cuda')


def _csr_matvec_scalar_gpu_translation(c, data, indices, indptr, vector, *, shape, transpose):
  if gpu_ops is None:
    raise GPUOperatorNotFound(_csrmv_scalar_p.name)

  if transpose:
    raise NotImplementedError

  data_shape = c.get_shape(data)
  type_name = b'float' if data_shape.element_type() == np.float32 else b'double'
  data_name = b'homo' if data_shape.dimensions() == (1,) else b'heter'
  opaque = gpu_ops.build_double_size_descriptor(shape[0], shape[1])
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'csr_matvec_' + data_name + b'_scalar_' + type_name,
    operands=(data, indices, indptr, vector),
    operand_shapes_with_layout=(c.get_shape(data),
                                c.get_shape(indices),
                                c.get_shape(indptr),
                                c.get_shape(vector)),
    shape_with_layout=xla_client.Shape.array_shape(data_shape.element_type(), (shape[0],), (0,)),
    opaque=opaque,
  )


def _csrmv_scalar_transpose(ct, data, indices, indptr, vector, *, shape, transpose):
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices.")

  if ad.is_undefined_primal(vector):
    ct_vector = _csrmv_scalar_p.bind(data, indices, indptr, ct, shape=shape, transpose=not transpose)
    return data, indices, indptr, (ad.Zero(vector) if type(ct) is ad.Zero else ct_vector)

  else:
    if type(ct) is ad.Zero:
      ct_data = ad.Zero(data)
    else:
      if data.aval.shape[0] == 1:  # scalar
        ct_data = _csrmv_scalar_p.bind(jnp.ones(1), indices, indptr, vector, shape=shape, transpose=transpose)
        ct_data = jnp.inner(ct, ct_data)
      else:  # heterogeneous values
        row, col = csr_to_coo(indices, indptr)
        ct_data = vector[row] * ct[col] if transpose else vector[col] * ct[row]
    return ct_data, indices, indptr, vector


_csrmv_scalar_p = core.Primitive('csr_matvec_scalar')
_csrmv_scalar_p.def_abstract_eval(_csrmv_abstract)
_csrmv_scalar_p.def_impl(partial(xla.apply_primitive, _csrmv_scalar_p))
xla.backend_specific_translations['cpu'][_csrmv_scalar_p] = _csrmv_cpu_translation
xla.backend_specific_translations['gpu'][_csrmv_scalar_p] = _csr_matvec_scalar_gpu_translation
ad.defjvp(_csrmv_scalar_p,
          partial(_csrmv_jvp_mat, _csrmv_scalar_p),
          None,
          None,
          partial(_csrmv_jvp_vec, _csrmv_scalar_p), )
ad.primitive_transposes[_csrmv_scalar_p] = _csrmv_scalar_transpose
register_general_batching(_csrmv_scalar_p)


def _csr_matvec_vector_gpu_translation(c, data, indices, indptr, vector, *, shape, transpose):
  if gpu_ops is None:
    raise GPUOperatorNotFound(_csrmv_vector_p.name)

  if transpose:
    raise NotImplementedError

  data_shape = c.get_shape(data)
  type_name = b'float' if data_shape.element_type() == jnp.float32 else b'double'
  data_name = b'homo' if data_shape.dimensions() == (1,) else b'heter'
  opaque = gpu_ops.build_double_size_descriptor(shape[0], shape[1])
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'csr_matvec_' + data_name + b'_vector_' + type_name,
    operands=(data, indices, indptr, vector),
    operand_shapes_with_layout=(c.get_shape(data),
                                c.get_shape(indices),
                                c.get_shape(indptr),
                                c.get_shape(vector)),
    shape_with_layout=xla_client.Shape.array_shape(data_shape.element_type(), (shape[0],), (0,)),
    opaque=opaque,
  )


def _csrmv_vector_transpose(ct, data, indices, indptr, vector, *, shape, transpose):
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices.")

  if ad.is_undefined_primal(vector):
    ct_vector = _csrmv_vector_p.bind(data, indices, indptr, ct, shape=shape, transpose=not transpose)
    return data, indices, indptr, (ad.Zero(vector) if type(ct) is ad.Zero else ct_vector)

  else:
    if type(ct) is ad.Zero:
      ct_data = ad.Zero(data)
    else:
      if data.aval.shape[0] == 1:  # scalar
        ct_data = _csrmv_vector_p.bind(jnp.ones(1), indices, indptr, vector, shape=shape, transpose=transpose)
        ct_data = jnp.inner(ct, ct_data)
      else:  # heterogeneous values
        row, col = csr_to_coo(indices, indptr)
        ct_data = vector[row] * ct[col] if transpose else vector[col] * ct[row]
    return ct_data, indices, indptr, vector


_csrmv_vector_p = core.Primitive('csr_matvec_vector')
_csrmv_vector_p.def_abstract_eval(_csrmv_abstract)
_csrmv_vector_p.def_impl(partial(xla.apply_primitive, _csrmv_vector_p))
xla.backend_specific_translations['cpu'][_csrmv_vector_p] = _csrmv_cpu_translation
xla.backend_specific_translations['gpu'][_csrmv_vector_p] = _csr_matvec_vector_gpu_translation
ad.defjvp(_csrmv_vector_p,
          partial(_csrmv_jvp_mat, _csrmv_vector_p),
          None,
          None,
          partial(_csrmv_jvp_vec, _csrmv_vector_p), )
ad.primitive_transposes[_csrmv_vector_p] = _csrmv_vector_transpose
register_general_batching(_csrmv_vector_p)


def _csr_matvec_adaptive_gpu_translation(c, data, indices, indptr, row_blocks, vector, *, shape, transpose):
  if gpu_ops is None:
    raise GPUOperatorNotFound(_csrmv_adaptive_p.name)

  if transpose:
    raise NotImplementedError

  data_shape = c.get_shape(data)
  type_name = b'float' if data_shape.element_type() == np.float32 else b'double'
  data_name = b'homo' if data_shape.dimensions() == (1,) else b'heter'
  opaque = gpu_ops.build_double_size_descriptor(shape[0], shape[1])
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'csr_matvec_' + data_name + b'_adaptive_' + type_name,
    operands=(data, indices, indptr, row_blocks, vector),
    operand_shapes_with_layout=(c.get_shape(data),
                                c.get_shape(indices),
                                c.get_shape(indptr),
                                c.get_shape(row_blocks),
                                c.get_shape(vector)),
    shape_with_layout=xla_client.Shape.array_shape(data_shape.element_type(), (shape[0],), (0,)),
    opaque=opaque,
  )


def _csrmv_adaptive_transpose(ct, data, indices, indptr, vector, *, shape, transpose):
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices.")

  if ad.is_undefined_primal(vector):
    ct_vector = _csrmv_adaptive_p.bind(data, indices, indptr, ct, shape=shape, transpose=not transpose)
    return data, indices, indptr, (ad.Zero(vector) if type(ct) is ad.Zero else ct_vector)

  else:
    if type(ct) is ad.Zero:
      ct_data = ad.Zero(data)
    else:
      if data.aval.shape[0] == 1:  # scalar
        ct_data = _csrmv_adaptive_p.bind(jnp.ones(1), indices, indptr, vector, shape=shape, transpose=transpose)
        ct_data = jnp.inner(ct, ct_data)
      else:  # heterogeneous values
        row, col = csr_to_coo(indices, indptr)
        ct_data = vector[row] * ct[col] if transpose else vector[col] * ct[row]
    return ct_data, indices, indptr, vector


_csrmv_adaptive_p = core.Primitive('csr_matvec_adaptive')
_csrmv_adaptive_p.def_abstract_eval(_csrmv_abstract)
_csrmv_adaptive_p.def_impl(partial(xla.apply_primitive, _csrmv_adaptive_p))
xla.backend_specific_translations['cpu'][_csrmv_adaptive_p] = _csrmv_cpu_translation
xla.backend_specific_translations['gpu'][_csrmv_adaptive_p] = _csr_matvec_adaptive_gpu_translation
ad.defjvp(_csrmv_adaptive_p,
          partial(_csrmv_jvp_mat, _csrmv_adaptive_p),
          None,
          None,
          partial(_csrmv_jvp_vec, _csrmv_adaptive_p), )
ad.primitive_transposes[_csrmv_adaptive_p] = _csrmv_adaptive_transpose
register_general_batching(_csrmv_adaptive_p)
