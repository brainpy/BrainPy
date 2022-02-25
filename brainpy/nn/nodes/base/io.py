# -*- coding: utf-8 -*-

from brainpy.nn.base import Node

__all__ = [
  'Input',
]


class Input(Node):
  def __init__(self, input_shape=None, name=None):
    super(Input, self).__init__(name=name, input_shape=input_shape)

  def ff_init(self):
    assert len(self.input_shapes) == 1, (f'{type(self).__name__} only support '
                                         f'receiving one feedforward input. ')
    self.set_output_shape(self.input_shapes[0])

  def call(self, ff, **kwargs):
    return ff[0]
