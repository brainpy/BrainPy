# -*- coding: utf-8 -*-

from brainpy import errors
from brainpy.base.base import Base

__all__ = [
  'Function',
]


class Function(Base):
  """The wrapper for Python functions.

  Parameters
  ----------
  f : function
    The function to wrap.
  nodes : list, tuple, dict
    The nodes in the defined function ``f``.
  name : optional, str
    The function name.
  """

  def __init__(self, f, nodes, name=None):
    # name
    self._f = f
    if name is None:
      name = self.unique_name(type=f.__name__ if hasattr(f, '__name__') else 'Function')

    # initialize
    super(Function, self).__init__(name=name)

    # nodes
    self._nodes = dict()
    if isinstance(nodes, Base):
      nodes = (nodes,)
    if isinstance(nodes, (tuple, list)):
      for i, node in enumerate(nodes):
        if not isinstance(node, Base):
          raise errors.BrainPyError(f'Must be an instance of {Base.__name__}, but we got {type(node)}.')
        self._nodes[f'_node{i}'] = node
    elif isinstance(nodes, dict):
      self._nodes.update(nodes)
    else:
      raise ValueError(f'Only support list/tuple/dict of {Base.__name__}, '
                       f'but we got {type(nodes)}: {nodes}')

  def nodes(self, method='absolute', _paths=None):
    if _paths is None:
      _paths = set()
    gather = self._nodes_in_container(self._nodes, method=method, _paths=_paths)
    gather.update(super(Function, self).nodes(method=method, _paths=_paths))
    return gather

  def __call__(self, *args, **kwargs):
    return self._f(*args, **kwargs)
