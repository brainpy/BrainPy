# -*- coding: utf-8 -*-


from brainpy import math, tools
from .base import TwoEndConnector


__all__ = [
  'MatConn',
  'IJConn',
]


class MatConn(TwoEndConnector):
  """Connector built from the connection matrix."""
  def __init__(self, conn_mat):
    super(MatConn, self).__init__()

    assert isinstance(conn_mat, math.ndarray) and conn_mat.ndim == 2
    self.conn_mat = math.asarray(conn_mat, dtype=math.bool_)
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

  def require(self, structures):
    self.check(structures)
    return self.returns(mat=self.conn_mat)


class IJConn(TwoEndConnector):
  """Connector built from the ``pre_ids`` and ``post_ids`` connections."""
  def __init__(self, i, j):
    super(IJConn, self).__init__()

    assert isinstance(i, math.ndarray) and i.ndim == 1
    assert isinstance(j, math.ndarray) and j.ndim == 1
    assert i.size == j.size

    # initialize the class via "pre_ids" and "post_ids"
    self.pre_ids = math.asarray(i, dtype=math.int_)
    self.post_ids = math.asarray(j, dtype=math.int_)

  def require(self, structures):
    self.check(structures)
    return self.returns(ij=(self.pre_ids, self.post_ids))
