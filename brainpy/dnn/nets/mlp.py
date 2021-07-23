# -*- coding: utf-8 -*-

from brainpy.dnn.initializers import XavierNormal, ZerosInit
from brainpy.dnn.layers import Linear, Sequential, Activation

__all__ = [
  'MLP'
]


class MLP(Sequential):
  def __init__(self, layer_sizes, activation='relu', w_init=XavierNormal(), b_init=ZerosInit(), name=None):
    assert len(layer_sizes) >= 2
    name = self.unique_name(name)
    ops = []
    for i in range(1, len(layer_sizes)):
      ops.append(Linear(n_in=layer_sizes[i - 1], n_out=layer_sizes[i],
                        w_init=w_init, b_init=b_init, name=f'{name}_l{i}'))
      ops.append(Activation(activation))
    super(MLP, self).__init__(*ops, name=name)
