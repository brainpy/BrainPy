# -*- coding: utf-8 -*-

from typing import Union, Sequence, Dict, Callable

from brainpy.base import FunAsObject, BrainPyObject
from brainpy.math.ndarray import Variable

__all__ = [
  'to_object',
]


def to_object(
    f: Callable = None,
    child_objs: Union[BrainPyObject, Sequence[BrainPyObject], Dict[str, BrainPyObject]] = None,
    dyn_vars: Union[Variable, Sequence[Variable], Dict[str, Variable]] = None,
    name: str = None
):
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
  func: FunAsObject
    The instance of ``BrainPyObject``.
  """

  if f is None:
    return lambda func: FunAsObject(f=func, child_objs=child_objs, dyn_vars=dyn_vars, name=name)

  else:
    if child_objs is None:
      raise ValueError(f'"nodes" cannot be None when "f" is provided.')
    return FunAsObject(f=f, child_objs=child_objs, dyn_vars=dyn_vars, name=name)
