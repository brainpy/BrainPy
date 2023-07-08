# -*- coding: utf-8 -*-

from brainpy._src.dynsys import IonChaDyn
from brainpy._src.mixin import TreeNode
from brainpy._src.dyn.ions.base import Calcium
from brainpy._src.dyn.neurons.hh import HHTypedNeuron

__all__ = [
  'IonChannel', 'IhChannel', 'CalciumChannel', 'SodiumChannel', 'PotassiumChannel', 'LeakyChannel',
]


class IonChannel(IonChaDyn, TreeNode):
  """Base class for ion channels."""

  '''The type of the master object.'''
  master_type = HHTypedNeuron

  def update(self, V):
    raise NotImplementedError('Must be implemented by the subclass.')

  def current(self, V):
    raise NotImplementedError('Must be implemented by the subclass.')

  def reset(self, V, batch_size=None):
    self.reset_state(V, batch_size)

  def reset_state(self, V, batch_size=None):
    raise NotImplementedError('Must be implemented by the subclass.')

  def clear_input(self):
    pass

  def __repr__(self):
    return f'{self.name}(size={self.size})'


class CalciumChannel(IonChannel):
  """Base class for Calcium ion channels."""

  master_type = Calcium
  '''The type of the master object.'''

  def update(self, V, C_Ca, E_Ca):
    raise NotImplementedError

  def current(self, V, C_Ca, E_Ca):
    raise NotImplementedError

  def reset(self, V, C_Ca, E_Ca, batch_size: int = None):
    self.reset_state(V, C_Ca, E_Ca, batch_size)

  def reset_state(self, V, C_Ca, E_Ca, batch_size: int = None):
    raise NotImplementedError('Must be implemented by the subclass.')


class IhChannel(IonChannel):
  """Base class for Ih channel models."""
  master_type = HHTypedNeuron


class PotassiumChannel(IonChannel):
  """Base class for potassium channel dynamics."""

  '''The type of the master object.'''
  master_type = HHTypedNeuron


class LeakyChannel(IonChannel):
  """Base class for leaky channel dynamics."""

  master_type = HHTypedNeuron


class SodiumChannel(IonChannel):
  """Base class for sodium channel dynamics."""

  master_type = HHTypedNeuron
