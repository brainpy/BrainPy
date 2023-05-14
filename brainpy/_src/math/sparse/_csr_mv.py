# -*- coding: utf-8 -*-


from functools import partial
from typing import Dict
from typing import Union, Tuple

import numba
import numpy as np
from jax import core, dtypes, default_backend
from jax import numpy as jnp, ensure_compile_time_eval
from jax import ops
from jax.core import ShapedArray
from jax.interpreters import ad, mlir, xla
from jax.lib import xla_client
from jaxlib import gpu_sparse

from brainpy._src import tools
from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import Array
from brainpy._src.math.op_registers import (compile_cpu_signature_with_numba,
                                            register_op_with_numba,
                                            register_general_batching)
from brainpy._src.math.sparse.utils import csr_to_coo
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
    version='cusparse',
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
  bl = tools.import_brainpylib()
  if version == 'cusparse':
    return bl.sparse_ops.cusparse_csr_matvec(data,
                                             indices,
                                             indptr,
                                             vector,
                                             shape=shape,
                                             transpose=transpose)
  elif version == 'vector':
    pass


def csr_matvec(
    data: Union[float, jnp.ndarray],
    indices: jnp.ndarray,
    indptr: jnp.ndarray,
    vector: jnp.ndarray,
    *,
    shape: Tuple[int, int],
    method: str = 'vector'
) -> jnp.ndarray:
  """CSR sparse matrix product with a dense vector, which outperforms the cuSPARSE algorithm.

  This function supports JAX transformations, including `jit()`,
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
  method: str
    The computing method used in GPU backend. Currently, we support
    `scalar`, `vector` and `adaptive`.

  Returns
  -------
  y : ndarry
    The array of shape ``(shape[1] if transpose else shape[0],)`` representing
    the matrix vector product.
  """
  bl = tools.import_brainpylib()
  return bl.sparse_ops.csr_matvec(data, indices, indptr, vector, shape=shape, method=method)


def _matmul_with_left_sparse(
    sparse: Dict,
    dense: Union[Array, jnp.ndarray]
):
  r"""Matrix multiplication with sparse matrix on the left.

  .. math::

    Y = M_{\mathrm{sparse}} @ M_{\mathrm{dense}}

  Parameters
  ----------
  sparse: dict
    The sparse matrix with shape of :math:`(N, M)`.
  dense: ArrayType
    The dense matrix with the shape of :math:`(M, K)`.

  Returns
  -------
  matrix
    A tensor the the shape of :math:`(N, K)`.
  """
  assert dense.ndim in [1, 2], 'Dense matrix must be a one- or two-dimensional matrix.'
  values = sparse['data']
  rows, cols = sparse['index']
  shape = sparse['shape']
  if len(shape) != 2:
    raise ValueError(f'Sparse matrix must be a two-dimensional matrix. But we got {shape}')
  values = as_jax(values)
  rows = as_jax(rows)
  cols = as_jax(cols)
  dense = as_jax(dense)
  B = dense.take(cols, axis=0)
  if B.ndim == 2:
    prod = B * jnp.reshape(values, (-1, 1))
  else:
    prod = B * values
  return ops.segment_sum(prod, rows, shape[0])


def _matmul_with_right_sparse(
    dense: Union[Array, jnp.ndarray],
    sparse: Dict
):
  r"""Matrix multiplication with sparse matrix on the left.

  .. math::

    Y = M_{\mathrm{dense}} @ M_{\mathrm{sparse}}

  Parameters
  ----------
  dense: ArrayType
    The dense matrix with the shape of :math:`(N, M)`.
  sparse: dict
    The sparse matrix with shape of :math:`(M, K)`.

  Returns
  -------
  matrix
    A tensor the the shape of :math:`(N, K)`.
  """
  assert dense.ndim in [1, 2], 'Dense matrix must be a one- or two-dimensional matrix.'
  values = sparse['data']
  rows, cols = sparse['index']
  shape = sparse['shape']
  if len(shape) != 2:
    raise ValueError(f'Sparse matrix must be a two-dimensional matrix. But we got {shape}')
  values = as_jax(values)
  rows = as_jax(rows)
  cols = as_jax(cols)
  dense = as_jax(dense)
  if dense.ndim == 2:
    A = dense[:, rows]
    prod = (A * values).T
    res = ops.segment_sum(prod, cols, shape[1]).T
  else:
    prod = dense[rows] * values
    res = ops.segment_sum(prod, cols, shape[1])
  return res


def sparse_matmul(A, B):
  r"""Sparse matrix multiplication.

  .. math::

     y = A @ B

  where :math:`A` or :math:`B` is a sparse matrix.
  :math:`A` and :math:`B` cannot be both sparse.

  Examples
  --------

  >>> import brainpy.math as bm

  1. when the left matrix :math:`A` is a sparse matrix with the shape of :math:`(N, M)`,

  >>> # A is a sparse matrix (3, 4):
  >>> #   [[0, 2, 0, 4],
  >>> #    [1, 0, 0, 0],
  >>> #    [0, 3, 0, 2]]
  >>> values = bm.asarray([2, 4, 1, 3, 2])
  >>> rows = bm.asarray([0, 0, 1, 2, 2])
  >>> cols = bm.asarray([1, 3, 0, 1, 3])
  >>> sparse = {'data': values, 'index': (rows, cols), 'shape': (3, 4)}
  >>> B = bm.arange(4)
  >>> bm.sparse_matmul(sparse, B)
  ArrayType([14,  0,  9], dtype=int32)
  >>> B = bm.random.rand(4, 3)
  >>> bm.sparse_matmul(sparse, B)
  ArrayType([[3.8331761 , 1.3708692 , 4.510223  ],
            [0.9960836 , 0.37550318, 0.7370341 ],
            [2.3700516 , 0.7574289 , 4.1124535 ]], dtype=float32)

  2. when the right matrix :math:`B` is a sparse matrix with the shape of :math:`(M, K)`,

  >>> A = bm.arange(3)
  >>> bm.sparse_matmul(A, sparse)
  ArrayType([1, 6, 0, 4], dtype=int32)
  >>> A = bm.random.rand(2, 3)
  >>> bm.sparse_matmul(A, sparse)
  ArrayType([[0.438388  , 1.4346815 , 0.        , 2.361964  ],
            [0.9171978 , 1.1214957 , 0.        , 0.90534496]],  dtype=float32)

  Parameters
  ----------
  A: tensor, sequence
    The dense or sparse matrix with the shape of :math:`(N, M)`.
  B: tensor, sequence
    The dense or sparse matrix with the shape of :math:`(M, K)`.

  Returns
  -------
  results: ArrayType
    The tensor with the shape of :math:`(N, K)`.
  """
  if isinstance(A, dict):
    if not isinstance(B, (Array, jnp.ndarray)):
      raise ValueError('A and B cannot be both sparse. \n'
                       f'A:\n{A}\n'
                       f'B:\n{B}')
    return _matmul_with_left_sparse(A, B)
  else:
    if not isinstance(B, dict):
      raise ValueError('A and B cannot be both dense. \n'
                       f'A:\n{A}\n'
                       f'B:\n{B}')
    return _matmul_with_right_sparse(A, B)


# --------------------------------------------------------------------
# cusparse_csr_matvec
# --------------------------------------------------------------------


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

  data = as_jax(data)
  indices = as_jax(indices)
  indptr = as_jax(indptr)
  vector = as_jax(vector)
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
  return cusparse_csr_matvec_p.bind(data, indices, indptr, vector, shape=shape, transpose=transpose)


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


cusparse_csr_matvec_p = core.Primitive('cusparse_csr_matvec')
cusparse_csr_matvec_p.def_abstract_eval(_csr_matvec_numba_abstract)
cusparse_csr_matvec_p.def_impl(partial(xla.apply_primitive, cusparse_csr_matvec_p))
xla.backend_specific_translations['cpu'][cusparse_csr_matvec_p] = _csr_matvec_cpu_translation
ad.defjvp(cusparse_csr_matvec_p, _csr_matvec_jvp_mat, None, None, _csr_matvec_jvp_vec)
ad.primitive_transposes[cusparse_csr_matvec_p] = _csr_matvec_transpose
register_general_batching(cusparse_csr_matvec_p)
if gpu_sparse.cuda_is_supported:
  mlir.register_lowering(cusparse_csr_matvec_p, _csr_matvec_gpu_lowering, platform='cuda')


def csr_matvec(
    data: Union[float, jnp.ndarray],
    indices: jnp.ndarray,
    indptr: jnp.ndarray,
    vector: jnp.ndarray,
    *,
    shape: Tuple[int, int],
    method: str = 'vector'
) -> jnp.ndarray:
  """CSR sparse matrix product with a dense vector, which outperforms the cuSPARSE algorithm.

  This function supports JAX transformations, including `jit()`,
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
    The computing method used in GPU backend. Currently, we support
    `scalar`, `vector` and `adaptive`.

  Returns
  -------
  y : ndarry
    The array of shape ``(shape[1] if transpose else shape[0],)`` representing
    the matrix vector product.
  """

  data = as_jax(data)
  indices = as_jax(indices)
  indptr = as_jax(indptr)
  vector = as_jax(vector)
  if method not in ['scalar', 'vector', 'adaptive']:
    raise ValueError('Only support methods: scalar, vector, and adaptive. '
                     f'But we got {method}.')

  data = jnp.atleast_1d(data)
  if not isinstance(data, jnp.ndarray):
    raise TypeError(f'data must a ndarray. But we got {type(data)}')
  if data.dtype not in [jnp.float32, jnp.float64]:
    raise TypeError(f'Only support float32 and float64. But we got {data.dtype}.')
  if data.dtype != vector.dtype:
    raise TypeError('The types of data and vector should be the same. '
                    f'But we got {data.dtype} != {vector.dtype}.')
  assert data.ndim == indices.ndim == indptr.ndim == vector.ndim == 1

  if method == 'adaptive':
    with ensure_compile_time_eval():
      pass
    return csr_matvec_adaptive_p.bind(data, indices, indptr, vector, shape=shape)

  elif method == 'scalar':
    return csr_matvec_scalar_p.bind(data, indices, indptr, vector, shape=shape)

  elif method == 'vector':
    return csr_matvec_vector_p.bind(data, indices, indptr, vector, shape=shape)

  else:
    raise ValueError('Only support methods: scalar, vector, and adaptive. '
                     f'But we got {method}.')


def _csr_matvec_abstract(*args, **kwargs):
  data = args[0]
  assert len(kwargs) == 1
  shape = kwargs['shape']
  return ShapedArray(dtype=data.dtype, shape=(shape[0],))


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _csr_matvec_numba_imp(outs, ins):
  data, indices, indptr, vector, shape = ins
  outs.fill(0)

  if len(data) == 1:
    data = data[0]
    for row_i in numba.prange(shape[0]):
      res = 0.
      for j in range(indptr[row_i], indptr[row_i + 1]):
        res += vector[indices[j]] * data
      outs[row_i] = res

  else:
    for row_i in numba.prange(shape[0]):
      res = 0.
      for j in range(indptr[row_i], indptr[row_i + 1]):
        res += vector[indices[j]] * data[j]
      outs[row_i] = res


def _csr_matvec_scalar_gpu_translation(c, data, indices, indptr, vector, *, shape):
  if gpu_ops is None:
    raise GPUOperatorNotFound(csr_matvec_scalar_p.name)

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


csr_matvec_scalar_p = register_op_with_numba(
  'csr_matvec_scalar',
  cpu_func=_csr_matvec_numba_imp,
  out_shapes=_csr_matvec_abstract,
  gpu_func_translation=_csr_matvec_scalar_gpu_translation,
)


def _csr_matvec_vector_gpu_translation(c, data, indices, indptr, vector, *, shape):
  if gpu_ops is None:
    raise GPUOperatorNotFound(csr_matvec_vector_p.name)

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


csr_matvec_vector_p = register_op_with_numba(
  'csr_matvec_vector',
  cpu_func=_csr_matvec_numba_imp,
  out_shapes=_csr_matvec_abstract,
  gpu_func_translation=_csr_matvec_vector_gpu_translation,
)


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _csr_matvec_adaptive(outs, ins):
  data, indices, indptr, vector, _, shape = ins
  outs.fill(0)

  if len(data) == 1:
    data = data[0]
    for i in numba.prange(shape[0]):
      res = 0.
      for j in range(indptr[i], indptr[i + 1]):
        res += vector[indices[j]] * data
      outs[i] = res

  else:
    for i in numba.prange(shape[0]):
      res = 0.
      for j in range(indptr[i], indptr[i + 1]):
        res += vector[indices[j]] * data[j]
      outs[i] = res


def _csr_matvec_adaptive_gpu_translation(c, data, indices, indptr, row_blocks, vector, *, shape):
  if gpu_ops is None:
    raise GPUOperatorNotFound(csr_matvec_adaptive_p.name)
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


csr_matvec_adaptive_p = register_op_with_numba(
  'csr_matvec_adaptive',
  out_shapes=_csr_matvec_abstract,
  cpu_func=_csr_matvec_adaptive,
  gpu_func_translation=_csr_matvec_adaptive_gpu_translation,
)
