# -*- coding: utf-8 -*-
import jax
import jax.numpy as jnp
import numpy as np

from brainpy import math as bm
from brainpy import tools
from brainpy.errors import ConnectorError
from .base import *

__all__ = [
  'MatConn',
  'IJConn',
  'CSRConn',
  'SparseMatConn'
]


class MatConn(TwoEndConnector):
  """Connector built from the dense connection matrix."""

  def __init__(self, conn_mat, **kwargs):
    super(MatConn, self).__init__(**kwargs)

    assert isinstance(conn_mat, (np.ndarray, bm.Array, jax.Array)) and conn_mat.ndim == 2
    self.pre_num, self.post_num = conn_mat.shape
    self.pre_size, self.post_size = (self.pre_num,), (self.post_num,)

    self.conn_mat = jnp.asarray(conn_mat).astype(MAT_DTYPE)

  def __call__(self, pre_size, post_size):
    assert self.pre_num == tools.size2num(pre_size)
    assert self.post_num == tools.size2num(post_size)
    return self

  def build_mat(self):
    assert self.conn_mat.shape[0] == self.pre_num
    assert self.conn_mat.shape[1] == self.post_num
    return self.conn_mat


class IJConn(TwoEndConnector):
  """Connector built from the ``pre_ids`` and ``post_ids`` connections."""

  def __init__(self, i, j, **kwargs):
    super(IJConn, self).__init__(**kwargs)

    assert isinstance(i, (np.ndarray, bm.Array, jnp.ndarray)) and i.ndim == 1
    assert isinstance(j, (np.ndarray, bm.Array, jnp.ndarray)) and j.ndim == 1
    assert i.size == j.size

    # initialize the class via "pre_ids" and "post_ids"
    self.pre_ids = jnp.asarray(i).astype(IDX_DTYPE)
    self.post_ids = jnp.asarray(j).astype(IDX_DTYPE)
    self.max_pre = self.pre_ids.max()
    self.max_post = self.post_ids.max()

  def __call__(self, pre_size, post_size):
    super(IJConn, self).__call__(pre_size, post_size)
    if self.max_pre >= self.pre_num:
      raise ConnectorError(f'pre_num ({self.pre_num}) should be greater than '
                           f'the maximum id ({self.max_pre}) of self.pre_ids.')
    if self.max_post >= self.post_num:
      raise ConnectorError(f'post_num ({self.post_num}) should be greater than '
                           f'the maximum id ({self.max_post}) of self.post_ids.')
    return self

  def build_coo(self):
    if self.pre_num <= self.max_pre:
      raise ConnectorError(f'pre_num ({self.pre_num}) should be greater than '
                           f'the maximum id ({self.max_pre}) of self.pre_ids.')
    if self.post_num <= self.max_post:
      raise ConnectorError(f'post_num ({self.post_num}) should be greater than '
                           f'the maximum id ({self.max_post}) of self.post_ids.')
    return self.pre_ids, self.post_ids


class CSRConn(TwoEndConnector):
  """Connector built from the CSR sparse connection matrix."""

  def __init__(self, indices, inptr, **kwargs):
    super(CSRConn, self).__init__(**kwargs)

    self.indices = jnp.asarray(indices, dtype=IDX_DTYPE)
    self.inptr = jnp.asarray(inptr, dtype=IDX_DTYPE)
    self.pre_num = self.inptr.size - 1
    self.max_post = self.indices.max()

  def build_csr(self):
    if self.pre_num != self.pre_num:
      raise ConnectorError(f'(pre_size, post_size) is inconsistent with '
                           f'the shape of the sparse matrix.')
    if self.post_num <= self.max_post:
      raise ConnectorError(f'post_num ({self.post_num}) should be greater than '
                           f'the maximum id ({self.max_post}) of self.post_ids.')
    return self.indices, self.inptr


class SparseMatConn(CSRConn):
  """Connector built from the sparse connection matrix"""

  def __init__(self, csr_mat, **kwargs):
    try:
      from scipy.sparse import csr_matrix
    except (ModuleNotFoundError, ImportError):
      raise ConnectorError(f'Using SparseMatConn requires the scipy package. '
                           f'Please run "pip install scipy" to install scipy.')

    assert isinstance(csr_mat, csr_matrix)
    self.csr_mat = csr_mat
    super(SparseMatConn, self).__init__(indices=jnp.asarray(self.csr_mat.indices, dtype=IDX_DTYPE),
                                        inptr=jnp.asarray(self.csr_mat.indptr, dtype=IDX_DTYPE),
                                        **kwargs)
    self.pre_num = csr_mat.shape[0]
    self.post_num = csr_mat.shape[1]
