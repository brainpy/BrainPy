# -*- coding: utf-8 -*-


from brainpy import math
from brainpy.primary import collector, checking

__all__ = [
  'Primary',
  'Function',
]


class Primary(object):
  def __init__(self, name=None):
    # check whether the object has a unique name.
    self.name = self.unique_name(name=name)
    checking.check_name(name=self.name, obj=self)

  def vars(self, method='absolute'):
    """Collect all the variables in the instance of DynamicSystem
    and the node instances.

    Parameters
    ----------
    method : str
      The prefix string for the variable names.

    Returns
    -------
    gather : datastructures.ArrayCollector
      The collection contained the variable name and the variable data.
    """
    gather = collector.ArrayCollector()
    if method == 'relative':
      for k, v in self.__dict__.items():
        if isinstance(v, math.ndarray):
          gather[k] = v
        elif isinstance(v, Primary):
          for k2, v2 in v.vars(method=method).items():
            gather[f'{k}.{k2}'] = v2
    elif method == 'absolute':
      for k, v in self.__dict__.items():
        if isinstance(v, math.ndarray):
          gather[f'{self.name}.{k}'] = v
        elif isinstance(v, Primary):
          gather.update(v.vars(method=method))
    else:
      raise ValueError(f'No support for the method of "{method}".')
    return gather

  def nodes(self, method='absolute'):
    """Collect all the nodes in the instance
    of DynamicSystem.

    Parameters
    ----------
    method : str
      The prefix string for the node names.

    Returns
    -------
    collector : collector.Collector
      The collection contained the integrator name and the integrator function.
    """
    gather = collector.Collector()
    if method == 'relative':
      for k, v in self.__dict__.items():
        if isinstance(v, Primary):
          gather[k] = v
          for k2, v2 in v.nodes(method=method).items():
            gather[f'{k}.{k2}'] = v2
    elif method == 'absolute':
      for k, v in self.__dict__.items():
        if isinstance(v, Primary):
          gather[v.name] = v
          gather.update(v.nodes(method=method))
    else:
      raise ValueError(f'No support for the method of "{method}".')
    return gather

  def unique_name(self, name=None, type=None):
    """Get the unique name for this object.

    Parameters
    ----------
    name : str, optional
      The expected name. If None, the unique name will be returned.
      Otherwise, the provided name will be checked to guarantee its
      uniqueness.
    type : str, optional
      The type of this class, used for object naming.

    Returns
    -------
    name : str
      The unique name for this object.
    """
    if name is None:
      if type is None:
        return checking.get_name(type=self.__class__.__name__)
      else:
        return checking.get_name(type=type)
    else:
      checking.check_name(name=name, obj=self)
      return name


class Function(Primary):
  def __init__(self, f, nodes, name=None):
    # name
    self._f = f
    if name is None:
      name = self.unique_name(type=f.__name__ if hasattr(f, '__name__') else 'Function')

    # initialize
    super(Function, self).__init__(name=name)

    # nodes
    self._nodes = dict()
    if isinstance(nodes, Primary):
      nodes = (nodes,)
    if isinstance(nodes, (tuple, list)):
      for i, node in enumerate(nodes):
        if not isinstance(node, Primary):
          raise ValueError
        self._nodes[f'_node{i}'] = node
    elif isinstance(nodes, dict):
      self._nodes.update(nodes)
    else:
      raise ValueError(f'Only support list/tuple/dict of {Primary.__name__}, '
                       f'but we got {type(nodes)}: {nodes}')

  def vars(self, method='absolute'):
    gather = collector.ArrayCollector()
    if method == 'relative':
      for i, (key, node) in enumerate(self._nodes.items()):
        for k, v in node.vars(method=method):
          gather[f'{key}.{k}'] = v
    elif method == 'absolute':
      for key, node in self._nodes.items():
        gather.update(node.vars(method=method))
    else:
      raise ValueError(f'No support for the method of "{method}".')
    return gather

  # def ints(self, method='absolute'):
  #   gather = collector.Collector()
  #   if method == 'relative':
  #     for i, (key, node) in enumerate(self._nodes.items()):
  #       for k, v in node.ints(method=method):
  #         gather[f'{key}.{k}'] = v
  #   elif method == 'absolute':
  #     for key, node in self._nodes.items():
  #       gather.update(node.ints(method=method))
  #   else:
  #     raise ValueError(f'No support for the method of "{method}".')
  #   return gather

  def nodes(self, method='absolute'):
    gather = collector.Collector()
    if method == 'relative':
      for i, (key, node) in enumerate(self._nodes.items()):
        gather[key] = node
        for k2, v2 in node.nodes(method=method):
          gather[f'{key}.{k2}'] = v2
    elif method == 'absolute':
      for key, node in self._nodes.items():
        gather[node.name] = node
        gather.update(node.nodes(method=method))
    else:
      raise ValueError(f'No support for the method of "{method}".')
    return gather

  def __call__(self, *args, **kwargs):
    return self._f(*args, **kwargs)
