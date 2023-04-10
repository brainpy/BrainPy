# -*- coding: utf-8 -*-


from typing import Tuple
from typing import Union, Dict

import jax.numpy as jnp
from jax import ops

from brainpy._src import tools
from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import Array

__all__ = [
  'cusparse_csr_matvec',
  'cusparse_coo_matvec',
  'csr_matvec',
  'sparse_matmul',
  'coo_to_csr',
  'csr_to_coo',
  'csr_to_dense'
]


def cusparse_csr_matvec(
    data: Union[float, jnp.ndarray, Array],
    indices: Union[jnp.ndarray, Array],
    indptr: Union[jnp.ndarray, Array],
    vector: Union[jnp.ndarray, Array],
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
  bl = tools.import_brainpylib()
  return bl.sparse_ops.cusparse_csr_matvec(data,
                                           indices,
                                           indptr,
                                           vector,
                                           shape=shape,
                                           transpose=transpose)


def cusparse_coo_matvec(
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
  bl = tools.import_brainpylib()
  return bl.sparse_ops.cusparse_coo_matvec(data,
                                           row,
                                           col,
                                           vector,
                                           shape=shape,
                                           rows_sorted=rows_sorted,
                                           cols_sorted=cols_sorted,
                                           transpose=transpose)


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


def coo_to_csr(
    pre_ids: jnp.ndarray,
    post_ids: jnp.ndarray,
    *,
    num_row: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """convert pre_ids, post_ids to (indices, indptr)."""
  bl = tools.import_brainpylib()
  return bl.sparse_ops.coo_to_csr(pre_ids, post_ids, num_row=num_row)


def csr_to_coo(
    indices: jnp.ndarray,
    indptr: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Given CSR (indices, indptr) return COO (row, col)"""
  bl = tools.import_brainpylib()
  return bl.sparse_ops.csr_to_coo(indices, indptr)


def csr_to_csc():
  pass


def coo_to_dense(
    data: jnp.ndarray,
    rows: jnp.ndarray,
    cols: jnp.ndarray,
    *,
    shape: Tuple[int, int]
) -> jnp.ndarray:
  pass


def csr_to_dense(
    data: jnp.ndarray,
    indices: jnp.ndarray,
    indptr: jnp.ndarray,
    *,
    shape: Tuple[int, int]
) -> jnp.ndarray:
  bl = tools.import_brainpylib()
  return bl.sparse_ops.csr_to_dense(data, indices, indptr, shape=shape)
