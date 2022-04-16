# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpy as np

from brainpy import tools
from brainpy.errors import ConnectorError
from brainpy.math.jaxarray import JaxArray
from .base import *

__all__ = [
  'MatConn',
  'IJConn',
  'SparseMatConn'
]


class MatConn(TwoEndConnector):
  """Connector built from the dense connection matrix."""

  def __init__(self, conn_mat):
    super(MatConn, self).__init__()

    assert isinstance(conn_mat, (np.ndarray, JaxArray, jnp.ndarray)) and conn_mat.ndim == 2
    self.pre_num, self.post_num = conn_mat.shape
    self.pre_size, self.post_size = (self.pre_num,), (self.post_num,)
    
    self.conn_mat = np.asarray(conn_mat, dtype=MAT_DTYPE)
  
  def __call__(self, pre_size, post_size):
    assert self.pre_num == tools.size2num(pre_size)
    assert self.post_num == tools.size2num(post_size)
    return self

  def build_conn(self):
    return 'mat', self.conn_mat


class IJConn(TwoEndConnector):
  """Connector built from the ``pre_ids`` and ``post_ids`` connections."""

  def __init__(self, i, j):
    super(IJConn, self).__init__()

    assert isinstance(i, (np.ndarray, JaxArray, jnp.ndarray)) and i.ndim == 1
    assert isinstance(j, (np.ndarray, JaxArray, jnp.ndarray)) and j.ndim == 1
    assert i.size == j.size

    # initialize the class via "pre_ids" and "post_ids"
    self.pre_ids = np.asarray(i, dtype=IDX_DTYPE)
    self.post_ids = np.asarray(j, dtype=IDX_DTYPE)

  def __call__(self, pre_size, post_size):
    super(IJConn, self).__call__(pre_size, post_size)

    max_pre = np.max(self.pre_ids)
    max_post = np.max(self.post_ids)
    if max_pre >= self.pre_num:
      raise ConnectorError(f'pre_num ({self.pre_num}) should be greater than '
                           f'the maximum id ({max_pre}) of self.pre_ids.')
    if max_post >= self.post_num:
      raise ConnectorError(f'post_num ({self.post_num}) should be greater than '
                           f'the maximum id ({max_post}) of self.post_ids.')
    return self

  def build_conn(self):
    return 'ij', (self.pre_ids, self.post_ids)


class SparseMatConn(TwoEndConnector):
  """Connector built from the sparse connection matrix"""

  def __init__(self, csr_mat):
    super(SparseMatConn, self).__init__()

    try:
      from scipy.sparse import csr_matrix
    except (ModuleNotFoundError, ImportError):
      raise ConnectorError(f'Using SparseMatConn requires the scipy package. '
                           f'Please run "pip install scipy" to install scipy.')

    assert isinstance(csr_mat, csr_matrix)
    csr_mat.data = np.asarray(csr_mat.data, dtype=MAT_DTYPE)
    self.csr_mat = csr_mat
    self.pre_num, self.post_num = csr_mat.shape

  def __call__(self, pre_size, post_size):
    try:
      assert self.pre_num == tools.size2num(pre_size)
      assert self.post_num == tools.size2num(post_size)
    except AssertionError:
      raise ConnectorError(f'(pre_size, post_size) is inconsistent with the shape of the sparse matrix.')

    super(SparseMatConn, self).__call__(pre_size, post_size)
    return self

  def build_conn(self):
    ind, indptr = self.csr_mat.indices, self.csr_mat.indptr
    return 'csr', (ind, indptr)
