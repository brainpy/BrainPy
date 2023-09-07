# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy._src.math.object_transform.base import BrainPyObject

__all__ = [
  'Encoder'
]


class Encoder(BrainPyObject):
  """Base class for encoding rate values as spike trains."""

  def __repr__(self):
    return self.__class__.__name__

  def single_step(self, *args, **kwargs):
    raise NotImplementedError('Please implement the function for single step encoding.')

  def multi_steps(self, *args, **kwargs):
    raise NotImplementedError('Encode implement the function for multiple-step encoding.')

