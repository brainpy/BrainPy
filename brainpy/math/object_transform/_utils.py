# -*- coding: utf-8 -*-

from brainpy.base import BrainPyObject, TensorCollector

__all__ = [
  'infer_dyn_vars'
]


def infer_dyn_vars(target):
  if isinstance(target, BrainPyObject):
    dyn_vars = target.vars().unique()
  elif hasattr(target, '__self__') and isinstance(target.__self__, BrainPyObject):
    dyn_vars = target.__self__.vars().unique()
  else:
    dyn_vars = TensorCollector()
  return dyn_vars
