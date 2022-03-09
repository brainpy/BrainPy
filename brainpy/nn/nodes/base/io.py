# -*- coding: utf-8 -*-

from typing import Tuple, Union

from brainpy.nn.base import Node

__all__ = [
  'Input',
]


class Input(Node):
  """The input node."""

  def __init__(self, input_shape: Union[Tuple[int], int] = None, name: str = None):
    super(Input, self).__init__(name=name, input_shape=input_shape)

  def ff_init(self):
    assert len(self.input_shapes) == 1, (f'{type(self).__name__} only support '
                                         f'receiving one feedforward input. ')
    self.set_output_shape(self.input_shapes[0])

  def forward(self, ff, **kwargs):
    return ff[0]
