# -*- coding: utf-8 -*-

from .base_object import BrainPyObject

__all__ = [
  'ObjectTransform'
]


class ObjectTransform(BrainPyObject):
  """Object-oriented JAX transformation for BrainPy computation.
  """
  def __init__(self, name: str = None):
    super().__init__(name=name)

  def __call__(self, *args, **kwargs):
    raise NotImplementedError

  def __repr__(self):
    return self.__class__.__name__
