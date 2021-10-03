# -*- coding: utf-8 -*-

from brainpy.simulation.initialize import XavierNormal, ZeroInit
from brainpy.simulation.layers.activation import Activation
from brainpy.simulation.layers.dense import Dense
from brainpy.simulation.layers.sequential import Sequential

__all__ = [
  'MLP'
]


class MLP(Sequential):
  """Multi-layer perceptron.
  """

  def __init__(self, layer_sizes, activation='relu', w_init=XavierNormal(), b_init=ZeroInit(), name=None):
    assert len(layer_sizes) >= 2
    name = self.unique_name(name)
    layers = []
    for i in range(1, len(layer_sizes)):
      layers.append(Dense(num_input=layer_sizes[i - 1], num_hidden=layer_sizes[i],
                          w_init=w_init, b_init=b_init, name=f'{name}_l{i}'))
      layers.append(Activation(activation))
    super(MLP, self).__init__(*layers, name=name)
