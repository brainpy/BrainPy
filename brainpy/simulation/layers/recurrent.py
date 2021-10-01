# -*- coding: utf-8 -*-

from brainpy import math

from brainpy.simulation.brainobjects.neuron import NeuGroup
from brainpy.simulation.brainobjects.synapse import TwoEndConn

__all__ = [
  'PlaceHolder',
  'RateRNN',
  'SpikingRNN',
  'Linear',
]


class PlaceHolder(NeuGroup):
  def __init__(self, size, **kwargs):
    super(PlaceHolder, self).__init__(size=size, **kwargs)

    self.s = math.zeros(size, dtype=math.float_)

  def update(self, _t, _dt):
    pass


class RateRNN(NeuGroup):
  def __init__(self, size, **kwargs):
    super(RateRNN, self).__init__(size=size, **kwargs)


class SpikingRNN(NeuGroup):
  def __init__(self, size, **kwargs):
    super(SpikingRNN, self).__init__(size=size, **kwargs)


class Linear(TwoEndConn):
  def __init__(self, pre, post, conn=None, **kwargs):
    super(Linear, self).__init__(pre, post, conn, **kwargs)
