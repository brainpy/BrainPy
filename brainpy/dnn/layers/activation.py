# -*- coding: utf-8 -*-

from brainpy.dnn import activations
from brainpy.dnn.base import Module

__all__ = [
  'Activation'
]


class Activation(Module):
  def __init__(self, activation, name=None, **setting):
    if name is None:
      name = self.unique_name(type=f'Activation_{activation}')
    super(Activation, self).__init__(name=name)

    self.activation = activations.get(activation)
    self.setting = setting

  def __call__(self, x):
    return self.activation(x, **self.setting)
