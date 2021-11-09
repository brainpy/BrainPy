# -*- coding: utf-8 -*-

import abc

from brainpy import math

__all__ = [
  'Initializer',
]


class Initializer(abc.ABC):
  """Base Initialization Class."""
  def __init__(self, dtype=None):
    self.dtype = math.float_ if dtype is None else dtype

  @abc.abstractmethod
  def __call__(self, shape):
    raise NotImplementedError
