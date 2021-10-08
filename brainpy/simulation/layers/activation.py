# -*- coding: utf-8 -*-

from brainpy import math
from brainpy.simulation.brainobjects.base import DynamicalSystem
from brainpy.simulation._imports import mjax

__all__ = [
  'Activation'
]


class Activation(DynamicalSystem):
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
      name = self.unique_name(type=f'Activation_{activation}')
    super(Activation, self).__init__(name=name)

    self._activation = mjax.activations.get(activation)
    self._setting = setting

  def update(self, x, **kwargs):
    return self._activation(x, **self._setting)
