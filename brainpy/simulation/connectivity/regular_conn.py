# -*- coding: utf-8 -*-

from brainpy import errors, math, tools
from brainpy.simulation import utils
from brainpy.simulation.connectivity.base import TwoEndConnector

__all__ = [
  'One2One', 'one2one',
  'All2All', 'all2all',
  'GridFour', 'grid_four',
  'GridEight', 'grid_eight',
  'GridN',
]


@tools.numba_jit
def _grid_four(height, width, row, include_self):
  conn_i = []
  conn_j = []

  for col in range(width):
    i_index = (row * width) + col
    if 0 <= row - 1 < height:
      j_index = ((row - 1) * width) + col
      conn_i.append(i_index)
      conn_j.append(j_index)
    if 0 <= row + 1 < height:
      j_index = ((row + 1) * width) + col
      conn_i.append(i_index)
      conn_j.append(j_index)
    if 0 <= col - 1 < width:
      j_index = (row * width) + col - 1
      conn_i.append(i_index)
      conn_j.append(j_index)
    if 0 <= col + 1 < width:
      j_index = (row * width) + col + 1
      conn_i.append(i_index)
      conn_j.append(j_index)
    if include_self:
      conn_i.append(i_index)
      conn_j.append(i_index)
  return conn_i, conn_j


@tools.numba_jit
def _grid_n(height, width, row, n, include_self):
  conn_i = []
  conn_j = []
  for col in range(width):
    i_index = (row * width) + col
    for row_diff in range(-n, n + 1):
      for col_diff in range(-n, n + 1):
        if (not include_self) and (row_diff == col_diff == 0):
          continue
        if 0 <= row + row_diff < height and 0 <= col + col_diff < width:
          j_index = ((row + row_diff) * width) + col + col_diff
          conn_i.append(i_index)
          conn_j.append(j_index)
  return conn_i, conn_j


class One2One(TwoEndConnector):
  """
  Connect two neuron groups one by one. This means
  The two neuron groups should have the same size.
  """

  def __init__(self):
    super(One2One, self).__init__()

  def __call__(self, pre_size, post_size):
    try:
      assert pre_size == post_size
    except AssertionError:
      raise errors.BrainPyError(f'One2One connection must be defined in two groups with the same size, '
                                f'but we got {pre_size} != {post_size}.')

    length = utils.size2len(pre_size)
    self.num_pre = length
    self.num_post = length

    self.pre_ids = math.arange(length, dtype=math.int_)
    self.post_ids = math.arange(length, dtype=math.int_)
    return self


one2one = One2One()


class All2All(TwoEndConnector):
  """Connect each neuron in first group to all neurons in the
  post-synaptic neuron groups. It means this kind of conn
  will create (num_pre x num_post) synapses.
  """

  def __init__(self, include_self=True):
    self.include_self = include_self
    super(All2All, self).__init__()

  def __call__(self, pre_size, post_size):
    pre_len = utils.size2len(pre_size)
    post_len = utils.size2len(post_size)
    self.num_pre = pre_len
    self.num_post = post_len

    mat = math.ones((pre_len, post_len), dtype=bool)
    if not self.include_self:
      mat = math.fill_diagonal(mat, False)
    pre_ids, post_ids = math.where(mat)
    self.pre_ids = math.asarray(pre_ids, dtype=math.int_)
    self.post_ids = math.asarray(post_ids, dtype=math.int_)
    self.conn_mat = math.asarray(mat, dtype=math.bool_)
    return self


all2all = All2All(include_self=True)


class GridFour(TwoEndConnector):
  """The nearest four neighbors conn method."""

  def __init__(self, include_self=False):
    super(GridFour, self).__init__()
    self.include_self = include_self

  def __call__(self, pre_size, post_size=None):
    self.num_pre = utils.size2len(pre_size)
    if post_size is not None:
      try:
        assert pre_size == post_size
      except AssertionError:
        raise errors.BrainPyError(f'The shape of pre-synaptic group should be the same with the '
                                  f'post group. But we got {pre_size} != {post_size}.')
      self.num_post = utils.size2len(post_size)
    else:
      self.num_post = self.num_pre

    if len(pre_size) == 1:
      height, width = pre_size[0], 1
    elif len(pre_size) == 2:
      height, width = pre_size
    else:
      raise errors.BrainPyError('Currently only support two-dimensional geometry.')
    conn_i = []
    conn_j = []
    for row in range(height):
      a = _grid_four(height, width, row, include_self=self.include_self)
      conn_i.extend(a[0])
      conn_j.extend(a[1])
    self.pre_ids = math.asarray(conn_i, dtype=math.int_)
    self.post_ids = math.asarray(conn_j, dtype=math.int_)
    return self


grid_four = GridFour()


class GridN(TwoEndConnector):
  """The nearest (2*N+1) * (2*N+1) neighbors conn method.

  Parameters
  ----------
  N : int
      Extend of the conn scope. For example:
      When N=1,
          [x x x]
          [x I x]
          [x x x]
      When N=2,
          [x x x x x]
          [x x x x x]
          [x x I x x]
          [x x x x x]
          [x x x x x]
  include_self : bool
      Whether create (i, i) conn ?
  """

  def __init__(self, N=1, include_self=False):
    super(GridN, self).__init__()
    self.N = N
    self.include_self = include_self

  def __call__(self, pre_size, post_size=None):
    self.num_pre = utils.size2len(pre_size)
    if post_size is not None:
      try:
        assert pre_size == post_size
      except AssertionError:
        raise errors.BrainPyError(
          f'The shape of pre-synaptic group should be the same with the post group. '
          f'But we got {pre_size} != {post_size}.')
      self.num_post = utils.size2len(post_size)
    else:
      self.num_post = self.num_pre

    if len(pre_size) == 1:
      height, width = pre_size[0], 1
    elif len(pre_size) == 2:
      height, width = pre_size
    else:
      raise errors.BrainPyError('Currently only support two-dimensional geometry.')

    conn_i = []
    conn_j = []
    for row in range(height):
      res = _grid_n(height=height, width=width, row=row,
                    n=self.N, include_self=self.include_self)
      conn_i.extend(res[0])
      conn_j.extend(res[1])
    self.pre_ids = math.asarray(conn_i, dtype=math.int_)
    self.post_ids = math.asarray(conn_j, dtype=math.int_)
    return self


class GridEight(GridN):
  """The nearest eight neighbors conn method."""

  def __init__(self, include_self=False):
    super(GridEight, self).__init__(N=1, include_self=include_self)


grid_eight = GridEight()
