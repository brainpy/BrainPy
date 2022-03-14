# -*- coding: utf-8 -*-

from typing import Optional, Callable
from brainpy import errors
from brainpy.base.base import Base
from brainpy.base import collector

math = None

__all__ = [
  'Function',
]


def _check_node(node):
  if not isinstance(node, Base):
    raise errors.BrainPyError(f'Element in "nodes" must be an instance of '
                              f'{Base.__name__}, but we got {type(node)}.')


def _check_var(var):
  global math
  if math is None: from brainpy import math
  if not isinstance(var, math.ndarray):
    raise errors.BrainPyError(f'Element in "dyn_vars" must be an instance of '
                              f'{math.ndarray.__name__}, but we got {type(var)}.')


class Function(Base):
  """The wrapper for Python functions.

  Parameters
  ----------
  f : function
    The function to wrap.
  nodes : optional, Base, sequence of Base, dict
    The nodes in the defined function ``f``.
  dyn_vars : optional, ndarray, sequence of ndarray, dict
    The dynamically changed variables.
  name : optional, str
    The function name.
  """

  def __init__(self, f: Optional[Callable], nodes=None, dyn_vars=None, name=None):
    # initialize 
    # ---
    self._f = f
    if name is None:
      name = self.unique_name(type_=f.__name__ if hasattr(f, '__name__') else 'Function')
    super(Function, self).__init__(name=name)

    # nodes 
    # ---
    if nodes is not None:
      self.implicit_nodes = collector.Collector()
      if isinstance(nodes, Base):
        nodes = (nodes,)
      if isinstance(nodes, (tuple, list)):
        for i, node in enumerate(nodes):
          _check_node(node)
          self.implicit_nodes[f'_node{i}'] = node
      elif isinstance(nodes, dict):
        for node in nodes.values():
          _check_node(node)
        self.implicit_nodes.update(nodes)
      else:
        raise ValueError(f'"nodes" only support list/tuple/dict of {Base.__name__}, '
                         f'but we got {type(nodes)}: {nodes}')

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
