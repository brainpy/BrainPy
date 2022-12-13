# -*- coding: utf-8 -*-

from brainpy.dyn.base import DynamicalSystem
from brainpy.modes import Mode, training

__all__ = [
  'Layer'
]


class Layer(DynamicalSystem):
  def __init__(self, name: str = None, mode: Mode = training):
    super().__init__(name=name, mode=mode)

  def reset_state(self, batch_size=None):
    pass

  def update(self, shr, x):
    raise NotImplementedError

