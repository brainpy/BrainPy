# -*- coding: utf-8 -*-

from brainpy.math import activations
from brainpy.rnn.base import Module

__all__ = [
  'Activation'
]


class Activation(Module):
  """Activation Layer.

  Parameters
  ----------
  activation : str
    The name of the activation function.
  name : optional, str
    The name of the class.
  setting : Any
    The settings for the activation function.
  """

  def __init__(self, activation, name=None, **setting):
    if name is None:
      name = self.unique_name(type_=f'Activation_{activation}')
    super(Activation, self).__init__(name=name)

    self._activation = activations.get(activation)
    self._setting = setting

  def update(self, x):
    return self._activation(x, **self._setting)
