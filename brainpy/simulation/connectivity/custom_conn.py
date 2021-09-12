# -*- coding: utf-8 -*-


from brainpy import math
from brainpy.simulation import utils
from brainpy.simulation.connectivity.base import TwoEndConnector


__all__ = [
  'MatConn',
  'IJConn',
]


class MatConn(TwoEndConnector):
  """Connector built from the connection matrix."""
  def __init__(self, conn_mat):
    super(MatConn, self).__init__()

    assert isinstance(conn_mat, math.ndarray) and conn_mat.ndim == 2
    self.conn_mat = conn_mat
    self.num_pre, self.num_post = conn_mat.shape

  def __call__(self, *args, **kwargs):
    return self


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

  def __call__(self, pre_size, post_size):
    # this is necessary when create "pre2post" ,
    # "pre2syn"  etc. structures
    self.num_pre = utils.size2len(pre_size)
    # this is necessary when create "post2pre" ,
    # "post2syn"  etc. structures
    self.num_post = utils.size2len(post_size)
    return self
