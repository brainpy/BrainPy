# -*- coding: utf-8 -*-
from abc import ABC

import numpy as np

import jax.numpy as jnp
from brainpy.math.jaxarray import JaxArray
from brainpy import tools, math
from .base import *
from scipy.sparse import csr_matrix

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
    self.conn_mat = np.asarray(conn_mat, dtype=MAT_DTYPE)
    self.pre_num, self.post_num = conn_mat.shape

  def __call__(self, pre_size, post_size):
    if isinstance(pre_size, int): pre_size = (pre_size,)
    pre_size = tuple(pre_size)
    if isinstance(post_size, int): post_size = (post_size,)
    post_size = tuple(post_size)
    self.pre_size, self.post_size = pre_size, post_size
    assert self.pre_num == tools.size2num(pre_size)
    assert self.post_num == tools.size2num(post_size)
    return self

  def require(self, *structures):
    self.check(structures)
    return self.make_return(mat=self.conn_mat)


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

  def require(self, *structures):
    self.check(structures)
    return self.make_return(ij=(self.pre_ids_list, self.post_ids_list))


class SparseMatConn(TwoEndConnector):
  """Connector built from the sparse connection matrix"""

  def __init__(self, csr_mat):
    super(SparseMatConn, self).__init__()

    assert isinstance(csr_mat, csr_matrix)
    self._data = csr_mat
    self.pre_num, self.post_num = csr_mat.shape

  def __call__(self, pre_size, post_size):
    if isinstance(pre_size, int): pre_size = (pre_size,)
    pre_size = tuple(pre_size)
    if isinstance(post_size, int): post_size = (post_size,)
    post_size = tuple(post_size)
    self.pre_size, self.post_size = pre_size, post_size
    assert self.pre_num == tools.size2num(pre_size)
    assert self.post_num == tools.size2num(post_size)
    return self

  def require(self, *structures):
    self.check(structures)
    return self.make_return(mat=self._data.todense())
