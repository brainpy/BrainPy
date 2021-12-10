# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

from brainpy import tools
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
    self.dense_mat = np.asarray(conn_mat, dtype=MAT_DTYPE)
    self.pre_num, self.post_num = conn_mat.shape

  def __call__(self, pre_size, post_size):
    assert self.pre_num == tools.size2num(pre_size)
    assert self.post_num == tools.size2num(post_size)
    self._reset_conn(pre_size=pre_size, post_size=post_size)
    self._data = csr_matrix(self.dense_mat)
    return self

class IJConn(TwoEndConnector):
  """Connector built from the ``pre_ids`` and ``post_ids`` connections."""

  def __init__(self, i, j):
    super(IJConn, self).__init__()

    assert isinstance(i, (np.ndarray, JaxArray, jnp.ndarray)) and i.ndim == 1
    assert isinstance(j, (np.ndarray, JaxArray, jnp.ndarray)) and j.ndim == 1
    assert i.size == j.size

    # initialize the class via "pre_ids" and "post_ids"
    self.pre_ids_list = np.asarray(i, dtype=IDX_DTYPE)
    self.post_ids_list = np.asarray(j, dtype=IDX_DTYPE)

  def __call__(self, pre_size, post_size):
    self._reset_conn(pre_size=pre_size, post_size=post_size)
    self._data = csr_matrix((np.ones_like(self.pre_ids_list, np.bool_), (self.pre_ids_list, self.post_ids_list)), shape=(pre_size, post_size))
    return self


class SparseMatConn(TwoEndConnector):
  """Connector built from the sparse connection matrix"""

  def __init__(self, csr_mat):
    super(SparseMatConn, self).__init__()

    assert isinstance(csr_mat, csr_matrix)
    self.csr_mat = csr_mat
    self.pre_num, self.post_num = csr_mat.shape

  def __call__(self, pre_size, post_size):
    assert self.pre_num == tools.size2num(pre_size)
    assert self.post_num == tools.size2num(post_size)
    self._reset_conn(pre_size=pre_size, post_size=post_size)
    self._data = self.csr_mat
    return self