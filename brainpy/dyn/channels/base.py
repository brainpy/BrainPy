# -*- coding: utf-8 -*-

from brainpy.dyn.base import Channel, CondNeuGroup

__all__ = [
  'Ion', 'IonChannel',
]


class Ion(Channel):
  """Base class for ions."""

  '''The type of the master object.'''
  master_type = CondNeuGroup

  def update(self, t, dt, V):
    raise NotImplementedError('Must be implemented by the subclass.')

  def reset(self, V):
    raise NotImplementedError('Must be implemented by the subclass.')

  def current(self, V):
    raise NotImplementedError('Must be implemented by the subclass.')

  def __repr__(self):
    return f'{self.__class__.__name__}(size={self.size})'


class IonChannel(Channel):
  """Base class for ion channels."""

  '''The type of the master object.'''
  master_type = CondNeuGroup

  def update(self, t, dt, V):
    raise NotImplementedError('Must be implemented by the subclass.')

  def current(self, V):
    raise NotImplementedError('Must be implemented by the subclass.')

  def reset(self, V):
    raise NotImplementedError('Must be implemented by the subclass.')

  def __repr__(self):
    return f'{self.__class__.__name__}(size={self.size})'
