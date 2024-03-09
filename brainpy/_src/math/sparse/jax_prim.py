from typing import Union, Dict

import jax.numpy as jnp
from jax import ops

from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import Array


__all__ = [
  'seg_matmul',
]


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


def seg_matmul(A, B):
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
  >>> bm.sparse.sparse_matmul(sparse, B)
  ArrayType([14,  0,  9], dtype=int32)
  >>> B = bm.random.rand(4, 3)
  >>> bm.sparse.sparse_matmul(sparse, B)
  ArrayType([[3.8331761 , 1.3708692 , 4.510223  ],
            [0.9960836 , 0.37550318, 0.7370341 ],
            [2.3700516 , 0.7574289 , 4.1124535 ]], dtype=float32)

  2. when the right matrix :math:`B` is a sparse matrix with the shape of :math:`(M, K)`,

  >>> A = bm.arange(3)
  >>> bm.sparse.sparse_matmul(A, sparse)
  ArrayType([1, 6, 0, 4], dtype=int32)
  >>> A = bm.random.rand(2, 3)
  >>> bm.sparse.sparse_matmul(A, sparse)
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


