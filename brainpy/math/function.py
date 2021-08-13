# -*- coding: utf-8 -*-

from brainpy import math
from brainpy.primary import Function

__all__ = [
  'function',
]


def function(f=None, nodes=None, name=None, jit=False):
  if f is None:
    if jit:
      return lambda func: math.jit(Function(f=func, nodes=nodes, name=name))
    else:
      return lambda func: Function(f=func, nodes=nodes, name=name)

  else:
    if nodes is None:
      raise ValueError(f'"nodes" cannot be None when "f" is provided.')
    if jit:
      return math.jit(Function(f=f, nodes=nodes, name=name))
    else:
      return Function(f=f, nodes=nodes, name=name)
