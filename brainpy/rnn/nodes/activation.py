# -*- coding: utf-8 -*-

from brainpy.math import activations
from brainpy.rnn.base import Node

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

  def __init__(self, activation, in_size=None, name=None, **fun_setting):
    if name is None:
      name = self.unique_name(type_=f'Activation_{activation}')
    super(Activation, self).__init__(in_size=in_size, name=name)

    self._activation = activations.get(activation)
    self._fun_setting = fun_setting

  def ff_init(self):
    self.set_out_size(self.in_size)

  def forward(self, x):
    return self._activation(x, **self._fun_setting)

  def reset(self, state=None):
    pass
