# -*- coding: utf-8 -*-

import abc

__all__ = [
  'Initializer',
  '_InterLayerInitializer',
  '_IntraLayerInitializer'
]


class Initializer(abc.ABC):
  """Base Initialization Class."""

  @abc.abstractmethod
  def __call__(self, shape, dtype=None):
    raise NotImplementedError

  def __repr__(self):
    return self.__class__.__name__


class _InterLayerInitializer(Initializer):
  """The superclass of Initializers that initialize the weights between two layers."""
  pass


class _IntraLayerInitializer(Initializer):
  """The superclass of Initializers that initialize the weights within a layer."""
  pass
