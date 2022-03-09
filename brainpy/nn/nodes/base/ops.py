# -*- coding: utf-8 -*-

from brainpy import math as bm, tools
from brainpy.nn import utils
from brainpy.tools.checking import check_shape_consistency
from brainpy.nn.base import Node

__all__ = [
  'Concat', 'Select', 'Reshape', 'Summation',
]


class Concat(Node):
  def __init__(self, axis=-1, **kwargs):
    super(Concat, self).__init__(**kwargs)
    self.axis = axis

  def ff_init(self):
    unique_shape, free_shapes = check_shape_consistency(self.input_shapes, self.axis)
    out_size = list(unique_shape)
    out_size.insert(self.axis, sum(free_shapes))
    self.set_output_shape(out_size)

  def forward(self, ff, **kwargs):
    return bm.concatenate(ff, axis=self.axis)


class Select(Node):
  def __init__(self, index, **kwargs):
    super(Select, self).__init__(**kwargs)
    if isinstance(index, int):
      self.index = bm.asarray([index]).value

  def ff_init(self):
    assert len(self.input_shapes) == 1, 'Only support select one Node.'
    out_size = bm.zeros(self.input_shapes[0])[self.index].shape
    self.set_output_shape(out_size)

  def forward(self, ff, **kwargs):
    ff = list(ff.values())[0]
    return ff[..., self.index]


class Reshape(Node):
  def __init__(self, shape, **kwargs):
    super(Reshape, self).__init__(**kwargs)
    self.shape = tools.to_size(shape)

  def ff_init(self):
    assert len(self.input_shapes) == 1, 'Only support reshape one Node.'
    in_size = self.input_shapes[0]
    if -1 in self.shape:
      length = bm.prod(in_size)
      out_size = list(self.shape)
      m1_idx = out_size.index(-1)
      other_shape = out_size[:m1_idx] + out_size[m1_idx + 1:]
      m1_length = int(length / bm.prod(other_shape))
      out_size[m1_idx] = m1_length
    else:
      assert bm.prod(in_size) == bm.prod(self.shape)
      out_size = self.shape
    self.set_output_shape(out_size)

  def forward(self, ff, **kwargs):
    return bm.reshape(ff[0], self.shape)


class Summation(Node):
  def __init__(self, **kwargs):
    super(Summation, self).__init__(**kwargs)

  def ff_init(self):
    unique_shape, _ = check_shape_consistency(self.input_shapes, None, True)
    self.set_output_shape(list(unique_shape))

  def forward(self, ff, **kwargs):
    res = ff[0]
    for v in ff[1:]:
      res = res + v
    return res
