# -*- coding: utf-8 -*-

from brainpy.math import activations
from brainpy.nn.base import Node

__all__ = [
  'Activation'
]


class Activation(Node):
  """Activation Layer.

  Parameters
  ----------
  activation : str
    The name of the activation function.
  name : optional, str
    The name of the class.
  fun_setting : Any
    The settings for the activation function.
  """

  def __init__(self, activation: str = 'relu', fun_setting=None, name=None, **kwargs):
    if name is None:
      name = self.unique_name(type_=f'{activation}_activation')
    super(Activation, self).__init__(name=name, **kwargs)

    self._activation = activations.get(activation)
    self._fun_setting = dict() if fun_setting is None else fun_setting

  def ff_init(self):
    assert len(self.input_shapes) == 1, f'{type(self).__name__} only support receiving one input. '
    self.set_output_shape(self.input_shapes[0])

  def call(self, ff, **kwargs):
    return self._activation(ff[0], **self._fun_setting)
