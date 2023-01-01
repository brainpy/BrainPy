# -*- coding: utf-8 -*-

from typing import Dict
from .base import BrainPyObject, ArrayCollector

__all__ = [
  'infer_dyn_vars',
  'get_brainpy_object',
]


def infer_dyn_vars(target):
  if isinstance(target, BrainPyObject):
    dyn_vars = target.vars().unique()
  elif hasattr(target, '__self__') and isinstance(target.__self__, BrainPyObject):
    dyn_vars = target.__self__.vars().unique()
  else:
    dyn_vars = ArrayCollector()
  return dyn_vars


def get_brainpy_object(target) -> Dict[str, BrainPyObject]:
  if isinstance(target, BrainPyObject):
    return {target.name: target}
  elif hasattr(target, '__self__') and isinstance(target.__self__, BrainPyObject):
    target = target.__self__
    return {target.name: target}
  else:
    return dict()
