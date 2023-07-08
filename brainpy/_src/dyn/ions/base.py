# -*- coding: utf-8 -*-

from typing import Union

import brainpy.math as bm
from brainpy._src.dyn.neurons.hh import CondNeuGroup
from brainpy._src.dynsys import IonChaDyn
from brainpy._src.mixin import Container, TreeNode
from brainpy.types import Shape

__all__ = [
  'Ion',
  'Calcium',
]


class Ion(IonChaDyn, TreeNode):
  """Base class for ions."""

  '''The type of the master object.'''
  master_type = CondNeuGroup

  def update(self, V):
    raise NotImplementedError('Must be implemented by the subclass.')

  def reset(self, V, batch_size=None):
    self.reset_state(V, batch_size)

  def reset_state(self, V, batch_size=None):
    raise NotImplementedError('Must be implemented by the subclass.')

  def current(self, V):
    raise NotImplementedError('Must be implemented by the subclass.')

  def clear_input(self):
    pass

  def __repr__(self):
    return f'{self.name}(size={self.size})'


class Calcium(Ion, Container):
  """The brainpy_object calcium dynamics.

  Parameters
  ----------
  size: int, sequence of int
    The size of the simulation target.
  method: str
    The numerical integration method.
  name: str
    The name of the object.
  **channels
    The calcium dependent channels.
  """

  '''The type of the master object.'''
  master_type = CondNeuGroup

  """Reversal potential."""
  E: Union[float, bm.Variable, bm.Array]

  """Calcium concentration."""
  C: Union[float, bm.Variable, bm.Array]

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      method: str = 'exp_auto',
      name: str = None,
      mode: bm.Mode = None,
      **channels
  ):
    super().__init__(size, keep_size=keep_size, mode=mode, method=method, name=name)

    self.children = bm.node_dict(self.format_elements(IonChaDyn, **channels))

  def update(self, V):
    for node in self.nodes(level=1, include_self=False).unique().subset(IonChaDyn).values():
      node.update(V, self.C, self.E)

  def current(self, V, C_Ca=None, E_Ca=None):
    C_Ca = self.C if (C_Ca is None) else C_Ca
    E_Ca = self.E if (E_Ca is None) else E_Ca
    nodes = tuple(self.nodes(level=1, include_self=False).unique().subset(IonChaDyn).values())

    if len(nodes) == 0:
      return 0.
    else:
      self.check_hierarchies(self.__class__, *nodes)
      current = nodes[0].current(V, C_Ca, E_Ca)
      for node in nodes[1:]:
        current += node.current(V, C_Ca, E_Ca)
      return current

