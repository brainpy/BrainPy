# -*- coding: utf-8 -*-

from brainpy import math as bm, tools
from ..base import Node

__all__ = [
  'Concat', 'Select', 'Reshape'
]


class Concat(Node):
  def __init__(self, axis=-1, name=None, in_size=None):
    super(Concat, self).__init__(name=name, in_size=in_size)

    self.axis = axis

  def ff_init(self):
    if self.has_feedback:
      raise ValueError('Do not support feedback. ')
    num_dim, other_dims, axis_dims = set(), set(), []
    for in_size in self.in_size:
      assert isinstance(in_size, (tuple, list))
      in_size = list(in_size)
      num_dim.add(len(in_size))
      axis_dims.append(in_size.pop(self.axis))
      other_dims.add(in_size)
    if len(num_dim) != 1:
      raise ValueError('Must be same shape.')
    if len(other_dims) != 1:
      raise ValueError('Other dimensions must be the same.')
    out_size = list(other_dims)[0]
    out_size.index(self.axis, sum(axis_dims))
    self.out_size = out_size

  def forward(self, x):
    return bm.concatenate(x, self.axis)


class Select(Node):
  def __init__(self, index, name=None, in_size=None):
    super(Select, self).__init__(name=name, in_size=in_size)

    if isinstance(index, int):
      self.index = bm.asarray([index])

  def ff_init(self):
    zeros = bm.zeros(self.in_size)
    try:
      out_size = zeros[self.index].shape
    except:
      raise ValueError(f'Cannot select elements at {self.index} '
                       f'with the input shape {self.in_size}.')
    self.out_size = out_size

  def forward(self, x):
    return x[self.index]


class Reshape(Node):
  def __init__(self, shape, name=None, in_size=None):
    super(Reshape, self).__init__(name=name, in_size=in_size)

    self.shape = tools.to_size(shape)

  def ff_init(self):
    zeros = bm.zeros(self.in_size)
    try:
      out_size = bm.reshape(zeros, self.shape).shape
    except:
      raise ValueError(f'Cannot reshape to {self.shape} with input size {self.in_size}. ')
    self.set_out_size(out_size)

  def forward(self, x):
    return bm.reshape(x, self.shape)
