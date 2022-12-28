# -*- coding: utf-8 -*-

import warnings
from typing import Union, Sequence, Dict, Callable

from .base_object import FunAsObject, BrainPyObject
from ..ndarray import Variable

__all__ = [
  'to_object',
  'to_dynsys',
  'function',
]


def to_object(
    f: Callable = None,
    child_objs: Union[Callable, BrainPyObject, Sequence[BrainPyObject], Dict[str, BrainPyObject]] = None,
    dyn_vars: Union[Variable, Sequence[Variable], Dict[str, Variable]] = None,
    name: str = None
):
  """Transform a Python function to :py:class:`~.BrainPyObject`.

  Parameters
  ----------
  f: function, callable
    The python function.
  child_objs: Callable, BrainPyObject, sequence of BrainPyObject, dict of BrainPyObject
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
    def wrap(func) -> FunAsObject:
      return FunAsObject(f=func, child_objs=child_objs, dyn_vars=dyn_vars, name=name)

    return wrap

  else:
    if child_objs is None:
      raise ValueError(f'"child_objs" cannot be None when "f" is provided.')
    return FunAsObject(f=f, child_objs=child_objs, dyn_vars=dyn_vars, name=name)


def to_dynsys(
    f: Callable = None,
    child_objs: Union[Callable, BrainPyObject, Sequence[BrainPyObject], Dict[str, BrainPyObject]] = None,
    dyn_vars: Union[Variable, Sequence[Variable], Dict[str, Variable]] = None,
    name: str = None
):
  """Transform a Python function to a :py:class:`~.DynamicalSystem`.

  Parameters
  ----------
  f: function, callable
    The python function.
  child_objs: Callable, DynamicalSystem, sequence of DynamicalSystem, dict of DynamicalSystem
    The children objects used in this Python function.
  dyn_vars: Variable, sequence of Variable, dict of Variable
    The `Variable` instance used in the Python function.
  name: str
    The name of the created ``BrainPyObject``.

  Returns
  -------
  func: FunAsDynSys
    The instance of ``DynamicalSystem``.
  """
  from brainpy.dyn.base import FuncAsDynSys

  if f is None:
    def wrap(func) -> FuncAsDynSys:
      return FuncAsDynSys(f=func, child_objs=child_objs, dyn_vars=dyn_vars, name=name)
    return wrap
  else:
    if child_objs is None:
      raise ValueError(f'"child_objs" cannot be None when "f" is provided.')
    return FuncAsDynSys(f=f, child_objs=child_objs, dyn_vars=dyn_vars, name=name)


def function(
    f: Callable = None,
    nodes: Union[Callable, BrainPyObject, Sequence[BrainPyObject], Dict[str, BrainPyObject]] = None,
    dyn_vars: Union[Variable, Sequence[Variable], Dict[str, Variable]] = None,
    name: str = None
):
  """Transform a Python function into a :py:class:`~.BrainPyObject`.

  .. deprecated:: 2.3.0
     Using :py:func:`~.to_object` instead.

  Parameters
  ----------
  f: function, callable
    The python function.
  nodes: Callable, BrainPyObject, sequence of BrainPyObject, dict of BrainPyObject
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
  warnings.warn('Using `brainpy.math.to_object()` instead. Will be removed after version 2.4.0.',
                UserWarning)
  return to_object(f, nodes, dyn_vars, name)
