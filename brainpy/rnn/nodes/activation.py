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
    in_sizes = list(self.in_size.values())
    assert len(in_sizes) == 1, f'{type(self).__name__} only support receiving one input. '
    self.set_out_size(in_sizes[0])

  def call(self, ff, fb=None):
    ff = list(ff.values())[0]
    return self._activation(ff, **self._fun_setting)
