# -*- coding: utf-8 -*-

from ..base import Node

__all__ = [
  'Input',
]


class Input(Node):
  def __init__(self, in_size=None, name=None):
    super(Input, self).__init__(name=name, in_size=in_size)

  def ff_init(self):
    self.set_out_size(self.in_size)

  def forward(self, x, y=None):
    return x

  def reset(self, state=None):
    pass

