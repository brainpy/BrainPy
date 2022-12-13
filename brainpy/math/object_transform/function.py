# -*- coding: utf-8 -*-

from brainpy.base import FunAsAObject

__all__ = [
  'to_object',
]


def to_object(f=None, child_objs=None, dyn_vars=None, name=None):
  """Transform a Python function to ``BrainPyObject``.

  Parameters
  ----------
  f: function, callable
    The python function.
  child_objs: BrainPyObject, sequence of BrainPyObject, dict of BrainPyObject
    The children objects used in this Python function.
  dyn_vars: Variable, sequence of Variable, dict of Variable
    The `Variable` instance used in the Python function.
  name: str
    The name of the created ``BrainPyObject``.

  Returns
  -------
  func: FunAsAObject
    The instance of ``BrainPyObject``.
  """

  if f is None:
    return lambda func: FunAsAObject(f=func, child_objs=child_objs, dyn_vars=dyn_vars, name=name)

  else:
    if child_objs is None:
      raise ValueError(f'"nodes" cannot be None when "f" is provided.')
    return FunAsAObject(f=f, child_objs=child_objs, dyn_vars=dyn_vars, name=name)
