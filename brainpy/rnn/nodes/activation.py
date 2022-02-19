# -*- coding: utf-8 -*-

from brainpy.math import activations
from brainpy.rnn.base_node import FeedForwardModule

__all__ = [
  'Activation'
]


class Activation(FeedForwardModule):
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

  def __init__(self, activation, name=None, **fun_setting):
    if name is None:
      name = self.unique_name(type_=f'Activation_{activation}')
    super(Activation, self).__init__(name=name)

    self._activation = activations.get(activation)
    self._fun_setting = fun_setting

  def init(self, x):
    self.in_size = x.shape
    self.out_size = x.shape

  def call(self, x):
    return self._activation(x, **self._fun_setting)

  def reset(self, state=None):
    pass
