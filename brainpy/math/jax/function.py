# -*- coding: utf-8 -*-

from brainpy.base.function import Function

__all__ = [
  'function',
]


def function(f=None, nodes=None, dyn_vars=None, name=None):
  if f is None:
    return lambda func: Function(f=func, nodes=nodes, dyn_vars=dyn_vars, name=name)

  else:
    if nodes is None:
      raise ValueError(f'"nodes" cannot be None when "f" is provided.')
    return Function(f=f, nodes=nodes, dyn_vars=dyn_vars, name=name)
