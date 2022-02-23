# -*- coding: utf-8 -*-

from brainpy import math as bm, tools
from brainpy.rnn import utils
from brainpy.rnn.base import Node

__all__ = [
  'Concat', 'Select', 'Reshape'
]


class Concat(Node):
  def __init__(self, axis=-1, **kwargs):
    super(Concat, self).__init__(**kwargs)
    self.axis = axis

  def ff_init(self):
    free_size, fixed_size = utils.check_shape(self.in_size, self.axis)
    out_size = list(fixed_size)
    out_size.insert(self.axis, sum(free_size))
    self.set_out_size(out_size)

  def call(self, ff, fb=None):
    values = list(ff.values())
    return bm.concatenate(values, axis=self.axis)


class Select(Node):
  def __init__(self, index, name=None, in_size=None):
    super(Select, self).__init__(name=name, in_size=in_size)
    if isinstance(index, int):
      self.index = bm.asarray([index])

  def ff_init(self):
    assert len(self.in_size) == 1, 'Only support select one Node.'
    in_size = list(self.in_size.items())[0]
    out_size = bm.zeros(in_size)[self.index].shape
    self.set_out_size(out_size)

  def call(self, ff, fb=None):
    ff = list(ff.values())[0]
    return ff[..., self.index]


class Reshape(Node):
  def __init__(self, shape, name=None, in_size=None):
    super(Reshape, self).__init__(name=name, in_size=in_size)
    self.shape = tools.to_size(shape)

  def ff_init(self):
    assert len(self.in_size) == 1, 'Only support reshape one Node.'
    in_size = list(self.in_size.items())[0]
    if -1 in self.shape:
      length = bm.prod(in_size)
      out_size = list(self.shape)
      m1_idx = out_size.index(-1)
      other_shape = out_size[:m1_idx] + out_size[m1_idx+1:]
      m1_length = int(length / bm.prod(other_shape))
      out_size[m1_idx] = m1_length
    else:
      assert bm.prod(in_size) == bm.prod(self.shape)
      out_size = self.shape
    self.set_out_size(out_size)

  def call(self, ff, fb=None):
    ff = list(ff.values())[0]
    return bm.reshape(ff, self.shape)
