# -*- coding: utf-8 -*-

from typing import Tuple, Union

from brainpy.nn.base import Node
from brainpy.nn.constants import PASS_ONLY_ONE
from brainpy.tools.others import to_size

__all__ = [
  'Input',
]


class Input(Node):
  """The input node."""

  data_pass_type = PASS_ONLY_ONE

  def __init__(self,
               input_shape: Union[Tuple[int], int],
               name: str = None):
    super(Input, self).__init__(name=name, input_shape=input_shape)
    self.set_feedforward_shapes({self.name: (None,) + to_size(input_shape)})
    self._init_ff()

  def init_ff(self):
    self.set_output_shape(self.feedforward_shapes)

  def forward(self, ff, **kwargs):
    return ff
