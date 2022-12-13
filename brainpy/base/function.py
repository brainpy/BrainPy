# -*- coding: utf-8 -*-

from typing import Optional, Callable
from brainpy import errors
from brainpy.base.base import BrainPyObject
from brainpy.base import collector

math = None

__all__ = [
  'FunAsAObject',
]


def _check_node(node):
  if not isinstance(node, BrainPyObject):
    raise errors.BrainPyError(f'Element in "nodes" must be an instance of '
                              f'{BrainPyObject.__name__}, but we got {type(node)}.')


def _check_var(var):
  global math
  if math is None: from brainpy import math
  if not isinstance(var, math.ndarray):
    raise errors.BrainPyError(f'Element in "dyn_vars" must be an instance of '
                              f'{math.ndarray.__name__}, but we got {type(var)}.')


class FunAsAObject(BrainPyObject):
  """The wrapper for Python functions.

  Parameters
  ----------
  f : function
    The function to wrap.
  child_objs : optional, BrainPyObject, sequence of BrainPyObject, dict
    The nodes in the defined function ``f``.
  dyn_vars : optional, ndarray, sequence of ndarray, dict
    The dynamically changed variables.
  name : optional, str
    The function name.
  """

  def __init__(self, f: Optional[Callable], child_objs=None, dyn_vars=None, name=None):
    # initialize 
    # ---
    self._f = f
    if name is None:
      name = self.unique_name(type_=f.__name__ if hasattr(f, '__name__') else 'FunAsAObject')
    super(FunAsAObject, self).__init__(name=name)

    # nodes 
    # ---
    if child_objs is not None:
      self.implicit_nodes = collector.Collector()
      if isinstance(child_objs, BrainPyObject):
        child_objs = (child_objs,)
      if isinstance(child_objs, (tuple, list)):
        for i, node in enumerate(child_objs):
          _check_node(node)
          self.implicit_nodes[f'_node{i}'] = node
      elif isinstance(child_objs, dict):
        for node in child_objs.values():
          _check_node(node)
        self.implicit_nodes.update(child_objs)
      else:
        raise ValueError(f'"child_objs" only support list/tuple/dict of {BrainPyObject.__name__}, '
                         f'but we got {type(child_objs)}: {child_objs}')

    # variables
    # ---
    if dyn_vars is not None:
      self.implicit_vars = collector.TensorCollector()
      global math
      if math is None: from brainpy import math
      if isinstance(dyn_vars, math.ndarray):
        dyn_vars = (dyn_vars,)
      if isinstance(dyn_vars, (tuple, list)):
        for i, v in enumerate(dyn_vars):
          _check_var(v)
          self.implicit_vars[f'_var{i}'] = v
      elif isinstance(dyn_vars, dict):
        for v in dyn_vars.values():
          _check_var(v)
        self.implicit_vars.update(dyn_vars)
      else:
        raise ValueError(f'"dyn_vars" only support list/tuple/dict of {math.ndarray.__name__}, '
                         f'but we got {type(dyn_vars)}: {dyn_vars}')

  def __call__(self, *args, **kwargs):
    return self._f(*args, **kwargs)
