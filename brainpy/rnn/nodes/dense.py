# -*- coding: utf-8 -*-


from typing import Dict, Optional

from brainpy import math as bm
from brainpy.initialize import XavierNormal, ZeroInit
from brainpy.types import Tensor
from brainpy.rnn.base import Node
from brainpy.rnn.utils import init_param, online, offline, check_shape, summation

__all__ = [
  'Dense', 'LinearReadout'
]


class Dense(Node):
  weights: Dict[Node, Tensor]
  bias: Optional[Dict[Node, Tensor]]

  def __init__(self, num_unit: int, w_init=XavierNormal(), b_init=ZeroInit(), **kwargs):
    super(Dense, self).__init__(**kwargs)
    self.num_unit = num_unit
    self.w_init = w_init
    self.b_init = b_init

  def ff_init(self):
    # initialize feedforward weights
    self.weights = dict()
    self.bias = dict()
    for node, in_size in self.in_size.items():  # 'in_size' may have batch information
      self.weights[node] = init_param(self.w_init, in_size[:-1] + (self.num_unit,))
      self.bias[node] = init_param(self.b_init, (self.num_unit,))
    if self.trainable:
      for node, bias in self.bias.items():
        self.weights[node] = bm.TrainVar(self.weights[node])
        self.bias[node] = None if bias is None else bm.TrainVar(bias)
    # set output size
    _, c_size = check_shape(self.in_size, -1)
    self.set_out_size(c_size + (self.num_unit,))

  def call(self, ff, fd=None):
    results = {}
    for node, tensor in ff.items():
      if self.bias[node] is None:
        results[node] = tensor @ self.weights[node]
      else:
        results[node] = tensor @ self.weights[node] + self.bias[node]
    return summation(results)

  @online
  def force_learning(self, x, y):
    pass

  @offline
  def ridge_regression(self):
    pass


class LinearReadout(Dense):
  def __init__(self, num_unit, w_init=XavierNormal(), b_init=ZeroInit(), **kwargs):
    super(LinearReadout, self).__init__(num_unit=num_unit, w_init=w_init, b_init=b_init, **kwargs)

  def ff_init(self):
    super(LinearReadout, self).ff_init()
    self.state = bm.Variable(bm.zeros(self.out_size, dtype=bm.float_))

  def call(self, ff, fd=None):
    self.state.value = super(LinearReadout, self).call(ff)
    return self.state
