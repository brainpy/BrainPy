# -*- coding: utf-8 -*-


import numpy as np

from brainpy import math as bm, tools
from brainpy.compat.nn.base import Node
from brainpy.compat.nn.datatypes import MultipleData
from brainpy.tools.checking import check_shape_consistency

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

  data_pass = MultipleData('sequence')

  def __init__(self, axis=-1, trainable=False, **kwargs):
    super(Concat, self).__init__(trainable=trainable, **kwargs)
    self.axis = axis

  def init_ff_conn(self):
    unique_shape, free_shapes = check_shape_consistency(self.feedforward_shapes, self.axis)
    out_size = list(unique_shape)
    out_size.insert(self.axis, sum(free_shapes))
    self.set_output_shape(out_size)

  def forward(self, ff, **shared_kwargs):
    return bm.concatenate(ff, axis=self.axis)


class Select(Node):
  """
  Select a subset of the given input.
  """

  def __init__(self, index, trainable=False, **kwargs):
    super(Select, self).__init__(trainable=trainable, **kwargs)
    if isinstance(index, int):
      self.index = bm.asarray([index]).value

  def init_ff_conn(self):
    out_size = bm.zeros(self.feedforward_shapes[1:])[self.index].shape
    self.set_output_shape((None,) + out_size)

  def forward(self, ff, **shared_kwargs):
    return ff[..., self.index]


class Reshape(Node):
  """
  Reshape the input tensor to another tensor.

  Parameters
  ----------
  shape: int, sequence of int
    The reshaped size. This shape does not contain the batch size.
  """

  def __init__(self, shape, trainable=False, **kwargs):
    super(Reshape, self).__init__(trainable=trainable, **kwargs)
    self.shape = tools.to_size(shape)
    assert (None not in self.shape), 'Batch size can not be defined in the reshaped size.'

  def init_ff_conn(self):
    in_size = self.feedforward_shapes[1:]
    if -1 in self.shape:
      assert self.shape.count(-1) == 1, f'Cannot set shape with multiple -1. But got {self.shape}'
      length = np.prod(in_size)
      out_size = list(self.shape)
      m1_idx = out_size.index(-1)
      other_shape = out_size[:m1_idx] + out_size[m1_idx + 1:]
      m1_length = int(length / np.prod(other_shape))
      out_size[m1_idx] = m1_length
    else:
      assert np.prod(in_size) == np.prod(self.shape)
      out_size = self.shape
    self.set_output_shape((None,) + tuple(out_size))

  def forward(self, ff, **shared_kwargs):
    return bm.reshape(ff, self.shape)


class Summation(Node):
  """
  Sum all input tensors into one.

  All inputs should be broadcast compatible.
  """
  data_pass = MultipleData('sequence')

  def __init__(self, trainable=False, **kwargs):
    super(Summation, self).__init__(trainable=trainable, **kwargs)

  def init_ff_conn(self):
    unique_shape, _ = check_shape_consistency(self.feedforward_shapes, None, True)
    self.set_output_shape(list(unique_shape))

  def forward(self, ff, **shared_kwargs):
    res = ff[0]
    for v in ff[1:]:
      res = res + v
    return res
