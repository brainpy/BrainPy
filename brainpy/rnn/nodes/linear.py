# -*- coding: utf-8 -*-


from brainpy import math as bm
from brainpy.initialize import XavierNormal, ZeroInit
from ..base import Node
from ..utils import init_param

__all__ = [
  'Dense', 'Readout'
]


class Dense(Node):
  def __init__(self, num_unit: int, in_size=None, name=None, trainable=False,
               W_init=XavierNormal(), b_init=ZeroInit(), ):
    super(Dense, self).__init__(name=name, in_size=in_size, trainable=trainable)
    self.num_unit = num_unit
    self.W_init = W_init
    self.b_init = b_init

  def ff_init(self):
    self.W = init_param(self.W_init, self.in_size[:-1] + (self.num_unit,))
    self.b = init_param(self.b_init, (self.num_unit,))
    self.set_out_size(self.in_size[:-1] + (self.num_unit,))
    if self.trainable:
      self.W = bm.TrainVar(self.W)
      self.b = bm.TrainVar(self.b)

  def forward(self, x):
    if self.b is None:
      return x @ self.W
    else:
      return x @ self.W + self.b


class Readout(Dense):
  def __init__(self, num_unit, in_size=None, name=None, trainable=False,
               W_init=XavierNormal(), b_init=ZeroInit()):
    super(Readout, self).__init__(num_unit=num_unit, name=name,
                                  in_size=in_size, trainable=trainable,
                                  W_init=W_init, b_init=b_init)

  def ff_init(self):
    super(Readout, self).ff_init()
    self.state = bm.Variable(bm.zeros(self.out_size, dtype=bm.float_))

  def forward(self, x):
    if self.b is None:
      self.state.value = x @ self.W
    else:
      self.state.value = x @ self.W + self.b
    return self.state
