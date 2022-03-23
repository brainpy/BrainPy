# -*- coding: utf-8 -*-

import abc

__all__ = [
  'Initializer',
  'InterLayerInitializer',
  'IntraLayerInitializer'
]


class Initializer(abc.ABC):
  """Base Initialization Class."""

  @abc.abstractmethod
  def __call__(self, shape, dtype=None):
    raise NotImplementedError


class InterLayerInitializer(Initializer):
  """The superclass of Initializers that initialize the weights between two layers."""

  @abc.abstractmethod
  def __call__(self, shape, dtype=None):
    raise NotImplementedError


class IntraLayerInitializer(Initializer):
  """The superclass of Initializers that initialize the weights within a layer."""

  @abc.abstractmethod
  def __call__(self, shape, dtype=None):
    raise NotImplementedError
