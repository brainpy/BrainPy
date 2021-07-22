# -*- coding: utf-8 -*-

from brainpy.simulation.brainobjects.function import Function

__all__ = [
  'function',
]


def function(f=None, nodes=None, name=None):
  if f is None:
    return lambda func: Function(f=func, nodes=nodes, name=name)

  else:
    if nodes is None:
      raise ValueError(f'"nodes" cannot be None when "f" is provided.')
    return Function(f=f, nodes=nodes, name=name)
