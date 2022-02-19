# -*- coding: utf-8 -*-

from brainpy import tools
from ..base_node import FeedForwardModule

__all__ = [
  'Input', 'Output'
]


class Input(FeedForwardModule):
  def __init__(self, size, name=None):
    super(Input, self).__init__(name=name)

    self._size = tools.to_size(size)

  def init(self, x=None):
    self.in_size = self._size
    self.out_size = self._size
    if x is not None:
      assert x.shape == self._size

  def call(self, x):
    return x

  def reset(self, state=None):
    pass


class Output(FeedForwardModule):
  def __init__(self, name=None):
    super(Output, self).__init__(name=name)

  def init(self, x=None):
    pass

  def call(self, x):
    return x

  def reset(self, state=None):
    pass
