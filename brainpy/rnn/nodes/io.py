# -*- coding: utf-8 -*-

from brainpy.rnn.base import Node

__all__ = [
  'Input',
]


class Input(Node):
  def __init__(self, in_size=None, name=None):
    super(Input, self).__init__(name=name, in_size=in_size)

  def ff_init(self):
    in_sizes = list(self.in_size.values())
    assert len(in_sizes) == 1, f'{type(self).__name__} only support receiving one input. '
    self.set_out_size(in_sizes[0])

  def call(self, ff, fb=None):
    return list(ff.values())[0]
