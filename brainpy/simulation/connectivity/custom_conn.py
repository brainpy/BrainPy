# -*- coding: utf-8 -*-


from brainpy import math
from brainpy.simulation.connectivity.base import TwoEndConnector


__all__ = [
  'MatConn',
  'IJConn',
]


class MatConn(TwoEndConnector):
  """Connector built from the connection matrix."""
  def __init__(self, mat):
    super(MatConn, self).__init__()

    assert isinstance(mat, math.ndarray) and mat.ndim == 2
    self.mat = math.asarray(mat, dtype=math.bool_)
    self.pre_num, self.post_num = mat.shape

  def __call__(self, *args, **kwargs):
    self.pre_size, self.post_size = self.pre_num, self.post_num
    return self

  def require(self, structures):
    self.check(structures)
    return self.returns(mat=self.mat)


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
