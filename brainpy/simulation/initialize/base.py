# -*- coding: utf-8 -*-

import abc

from brainpy import math

__all__ = [
  'Initializer',
]


class Initializer(abc.ABC):
  """Base Initialization Class."""
  @abc.abstractmethod
  def __call__(self, shape):
    raise NotImplementedError
