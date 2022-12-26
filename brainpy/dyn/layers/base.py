# -*- coding: utf-8 -*-


from typing import Optional, Any

import brainpy.math as bm
from brainpy.dyn.base import DynamicalSystem

__all__ = [
  'Layer'
]


class Layer(DynamicalSystem):
  """Base class for a layer of artificial neural network."""

  def __init__(self,
               name: Optional[str] = None,
               mode: Optional[bm.Mode] = None):
    super().__init__(name=name, mode=mode)

  def reset_state(self, batch_size: Optional[int] = None):
    pass

  def clear_input(self):
    pass

  def update(self, shr: dict, x: Any):
    raise NotImplementedError
