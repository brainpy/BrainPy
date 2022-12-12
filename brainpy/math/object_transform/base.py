# -*- coding: utf-8 -*-

from brainpy.base.base import Base

__all__ = [
  'ObjectTransform'
]


class ObjectTransform(Base):
  """Object-oriented JAX transformation for BrainPy computation.
  """
  def __init__(self, name: str = None):
    super().__init__(name=name)

  def __call__(self, *args, **kwargs):
    raise NotImplementedError
