# -*- coding: utf-8 -*-

from brainpy.base import collector
from brainpy.base.base import Base

__all__ = [
  'Function',
]


class Function(Base):
  """The wrapper for Python functions."""

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
          raise ValueError
        self._nodes[f'_node{i}'] = node
    elif isinstance(nodes, dict):
      self._nodes.update(nodes)
    else:
      raise ValueError(f'Only support list/tuple/dict of {Base.__name__}, '
                       f'but we got {type(nodes)}: {nodes}')

  def vars(self, method='absolute'):
    gather = collector.ArrayCollector()
    if method == 'absolute':
      for key, node in self._nodes.items():
        gather.update(node.vars(method=method))
    elif method == 'relative':
      for i, (key, node) in enumerate(self._nodes.items()):
        for k, v in node.vars(method=method):
          gather[f'{key}.{k}'] = v
    else:
      raise ValueError(f'No support for the method of "{method}".')
    return gather

  def ints(self, method='absolute'):
    gather = collector.Collector()
    if method == 'absolute':
      for key, node in self._nodes.items():
        gather.update(node.ints(method=method))
    elif method == 'relative':
      for key, node in self._nodes.items():
        for k, v in node.ints(method=method):
          gather[f'{key}.{k}'] = v
    else:
      raise ValueError(f'No support for the method of "{method}".')
    return gather

  def nodes(self, method='absolute'):
    gather = collector.Collector()
    if method == 'absolute':
      for key, node in self._nodes.items():
        gather[node.name] = node
        gather.update(node.nodes(method=method))
    elif method == 'relative':
      for key, node in self._nodes.items():
        gather[key] = node
        for k2, v2 in node.nodes(method=method):
          gather[f'{key}.{k2}'] = v2
    else:
      raise ValueError(f'No support for the method of "{method}".')
    return gather

  def __call__(self, *args, **kwargs):
    return self._f(*args, **kwargs)
