# -*- coding: utf-8 -*-

from brainpy import math as bm, tools
from brainpy.nn.constants import PASS_ONLY_ONE
from brainpy.tools.checking import check_shape_consistency
from brainpy.nn.base import Node

__all__ = [
  'Concat', 'Select', 'Reshape', 'Summation',
]


class Concat(Node):
  """
  Concatenate multiple inputs into one.

  Parameters
  ----------
  axis : int
    The axis of concatenation to perform.
  """
  def __init__(self, axis=-1, **kwargs):
    super(Concat, self).__init__(**kwargs)
    self.axis = axis

  def init_ff(self):
    unique_shape, free_shapes = check_shape_consistency(self.input_shapes, self.axis)
    out_size = list(unique_shape)
    out_size.insert(self.axis, sum(free_shapes))
    self.set_output_shape(out_size)

  def forward(self, ff, **kwargs):
    return bm.concatenate(ff, axis=self.axis)


class Select(Node):
  """
  Select a subset of the given input.
  """

  data_pass_type = PASS_ONLY_ONE

  def __init__(self, index, **kwargs):
    super(Select, self).__init__(**kwargs)
    if isinstance(index, int):
      self.index = bm.asarray([index]).value

  def init_ff(self):
    out_size = bm.zeros(self.input_shapes[1:])[self.index].shape
    self.set_output_shape((None, ) + out_size)

  def forward(self, ff, **kwargs):
    return ff[..., self.index]


class Reshape(Node):
  """
  Reshape the input tensor to another tensor.

  Parameters
  ----------
  shape: int, sequence of int
    The reshaped size. This shape does not contain the batch size.
  """
  data_pass_type = PASS_ONLY_ONE

  def __init__(self, shape, **kwargs):
    super(Reshape, self).__init__(**kwargs)
    self.shape = tools.to_size(shape)
    assert (None not in self.shape), 'Batch size can not be defined in the reshaped size.'

  def init_ff(self):
    in_size = self.input_shapes[1:]
    if -1 in self.shape:
      assert self.shape.count(-1) == 1, f'Cannot set shape with multiple -1. But got {self.shape}'
      length = bm.prod(in_size)
      out_size = list(self.shape)
      m1_idx = out_size.index(-1)
      other_shape = out_size[:m1_idx] + out_size[m1_idx + 1:]
      m1_length = int(length / bm.prod(other_shape))
      out_size[m1_idx] = m1_length
    else:
      assert bm.prod(in_size) == bm.prod(self.shape)
      out_size = self.shape
    self.set_output_shape((None, ) + out_size)

  def forward(self, ff, **kwargs):
    return bm.reshape(ff, self.shape)


class Summation(Node):
  """
  Sum all input tensors into one.

  All inputs should be broadcast compatible.
  """
  def __init__(self, **kwargs):
    super(Summation, self).__init__(**kwargs)

  def init_ff(self):
    unique_shape, _ = check_shape_consistency(self.input_shapes, None, True)
    self.set_output_shape(list(unique_shape))

  def forward(self, ff, **kwargs):
    res = ff[0]
    for v in ff[1:]:
      res = res + v
    return res
