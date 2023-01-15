# -*- coding: utf-8 -*-


from typing import Optional

import brainpy.math as bm
from brainpy._src.dyn.base import DynamicalSystem

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
    child_nodes = self.nodes(level=1, include_self=False).subset(DynamicalSystem).unique()
    if len(child_nodes) > 0:
      for node in child_nodes.values():
        node.reset_state(batch_size=batch_size)
      self.reset_local_delays(child_nodes)
    else:
      pass

  def clear_input(self):
    child_nodes = self.nodes(level=1, include_self=False).subset(DynamicalSystem).unique()
    if len(child_nodes) > 0:
      for node in child_nodes.values():
        node.clear_input()
    else:
      pass

  def update(self, *args):
    raise NotImplementedError
