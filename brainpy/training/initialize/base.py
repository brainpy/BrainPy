# -*- coding: utf-8 -*-

import abc
from abc import ABC

from brainpy import math

__all__ = [
  'Initializer',
  'InterLayerInitializer',
  'IntraLayerInitializer'
]


class Initializer(abc.ABC):
  """Base Initialization Class."""

  @abc.abstractmethod
  def __call__(self, shape):
    raise NotImplementedError


class InterLayerInitializer(Initializer):
  """The superclass of Initializers that initialize the weights between two layers."""

  @abc.abstractmethod
  def __call__(self, shape):
    raise NotImplementedError


class IntraLayerInitializer(Initializer):
  """The superclass of Initializers that initialize the weights within a layer."""

  @abc.abstractmethod
  def __call__(self, shape):
    raise NotImplementedError
