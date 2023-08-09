# -*- coding: utf-8 -*-

from brainpy._src.dyn.base import IonChaDyn
from brainpy._src.mixin import TreeNode
from brainpy._src.dyn.neurons.hh import HHTypedNeuron

__all__ = [
  'IonChannel',
]


class IonChannel(IonChaDyn, TreeNode):
  """Base class for ion channels."""

  '''The type of the master object.'''
  master_type = HHTypedNeuron

  def update(self, *args, **kwargs):
    raise NotImplementedError('Must be implemented by the subclass.')

  def current(self, *args, **kwargs):
    raise NotImplementedError('Must be implemented by the subclass.')

  def reset_state(self, *args, **kwargs):
    raise NotImplementedError('Must be implemented by the subclass.')

  def clear_input(self):
    pass

  def __repr__(self):
    return f'{self.name}(size={self.size})'
