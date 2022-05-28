# -*- coding: utf-8 -*-

from typing import Tuple, Union

from brainpy.nn.base import Node
from brainpy.tools.others import to_size

__all__ = [
  'Input',
]


class Input(Node):
  """The input node."""

  def __init__(
      self,
      input_shape: Union[Tuple[int, ...], int],
      trainable: bool = False,
      name: str = None,
  ):
    super(Input, self).__init__(name=name, trainable=trainable, input_shape=input_shape)
    self.set_feedforward_shapes({self.name: (None,) + to_size(input_shape)})
    self._init_ff_conn()

  def init_ff_conn(self):
    self.set_output_shape(self.feedforward_shapes)

  def forward(self, ff, **shared_kwargs):
    return ff
