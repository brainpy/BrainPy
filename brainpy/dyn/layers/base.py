# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.dyn.base import DynamicalSystem

__all__ = [
  'Layer'
]


class Layer(DynamicalSystem):
  def __init__(self, name: str = None, mode: bm.Mode = None):
    super().__init__(name=name, mode=mode)

  def reset_state(self, batch_size=None):
    pass

  def update(self, shr, x):
    raise NotImplementedError

