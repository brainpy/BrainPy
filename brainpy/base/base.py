# -*- coding: utf-8 -*-


from brainpy.base import collector
from brainpy.tools import namechecking

__all__ = [
  'Base',
]

math = None
DE_INT = None


class Base(object):
  """The Base class for whole BrainPy ecosystem.

  The subclass of Base includes:

  - ``Module`` in brainpy.dnn.base.py
  - ``DynamicSystem`` in brainpy.simulation.brainobjects.base.py

  """

  def __init__(self, name=None):
    # check whether the object has a unique name.
    self.name = self.unique_name(name=name)
    namechecking.check_name(name=self.name, obj=self)

  def vars(self, method='absolute'):
    """Collect all variables in this node and the children nodes.

    Parameters
    ----------
    method : str
      The method to access the variables.

    Returns
    -------
    gather : collector.ArrayCollector
      The collection contained (the path, the variable).
    """
    global math
    if math is None:
      from brainpy import math

    gather = collector.ArrayCollector()
    if method == 'absolute':
      for k, v in self.__dict__.items():
        if isinstance(v, math.Variable):
          gather[f'{self.name}.{k}'] = v
        elif isinstance(v, Base):
          gather.update(v.vars(method=method))
    elif method == 'relative':
      for k, v in self.__dict__.items():
        if isinstance(v, math.Variable):
          gather[k] = v
        elif isinstance(v, Base):
          for k2, v2 in v.vars(method=method).items():
            gather[f'{k}.{k2}'] = v2
    else:
      raise ValueError(f'No support for the method of "{method}".')
    return gather

  def _vars_in_container(self, dict_container, method='absolute'):
    gather = collector.ArrayCollector()
    if method == 'absolute':
      for _, v in dict_container.items():
        gather.update(v.vars(method=method))
    elif method == 'relative':
      for k, v in dict_container.items():
        for k2, v2 in v.vars(method=method).items():
          gather[f'{k}.{k2}'] = v2
    else:
      raise ValueError(f'No support for the method of "{method}".')
    return gather

  def train_vars(self, method='absolute'):
    """The shortcut for retrieving all trainable variables.

    Parameters
    ----------
    method : str
      The method to access the variables. Support 'absolute' and 'relative'.

    Returns
    -------
    gather : collector.ArrayCollector
      The collection contained (the path, the trainable variable).
    """
    global math
    if math is None:
      from brainpy import math
    return self.vars(method=method).subset(math.TrainVar)

  def nodes(self, method='absolute', _paths=None):
    """Collect all children nodes.

    Parameters
    ----------
    method : str
      The method to access the nodes.
    _paths : set, Optional
      The data structure to solve the circular reference.

    Returns
    -------
    gather : collector.Collector
      The collection contained (the path, the node).
    """
    if _paths is None:
      _paths = set()
    gather = collector.Collector()
    if method == 'absolute':
      nodes = []
      for k, v in self.__dict__.items():
        if isinstance(v, Base):
          path = (id(self), id(v))
          if path not in _paths:
            _paths.add(path)
            gather[v.name] = v
            nodes.append(v)
      for v in nodes:
        gather.update(v.nodes(method=method, _paths=_paths))
    elif method == 'relative':
      nodes = []
      for k, v in self.__dict__.items():
        if isinstance(v, Base):
          path = (id(self), id(v))
          if path not in _paths:
            _paths.add(path)
            gather[k] = v
            nodes.append((k, v))
      for k, v in nodes:
        for k2, v2 in v.nodes(method=method, _paths=_paths).items():
          gather[f'{k}.{k2}'] = v2
    else:
      raise ValueError(f'No support for the method of "{method}".')
    return gather

  def _nodes_in_container(self, dict_container, method='absolute', _paths=None):
    if _paths is None:
      _paths = set()

    gather = collector.Collector()
    if method == 'absolute':
      nodes = []
      for _, node in dict_container.items():
        path = (id(self), id(node))
        if path not in _paths:
          _paths.add(path)
          gather[node.name] = node
          nodes.append(node)
      for node in nodes:
        gather[node.name] = node
        gather.update(node.nodes(method=method, _paths=_paths))
    elif method == 'relative':
      nodes = []
      for key, node in dict_container.items():
        path = (id(self), id(node))
        if path not in _paths:
          _paths.add(path)
          gather[key] = node
          nodes.append((key, node))
      for key, node in nodes:
        for key2, node2 in node.nodes(method=method, _paths=_paths).items():
          gather[f'{key}.{key2}'] = node2
    else:
      raise ValueError(f'No support for the method of "{method}".')
    return gather

  def ints(self, method='absolute'):
    """Collect all integrators in this node and the children nodes.

    Parameters
    ----------
    method : str
      The method to access the integrators.

    Returns
    -------
    collector : collector.Collector
      The collection contained (the path, the integrator).
    """
    global DE_INT
    if DE_INT is None:
      from brainpy.integrators.constants import DE_INT

    nodes = self.nodes(method=method)
    gather = collector.Collector()
    for node_path, node in nodes.items():
      for k in dir(node):
        v = getattr(node, k)
        if callable(v) and hasattr(v, '__name__') and v.__name__.startswith(DE_INT):
          gather[f'{node_path}.{k}'] = v
    return gather

  def unique_name(self, name=None, type=None):
    """Get the unique name for this object.

    Parameters
    ----------
    name : str, optional
      The expected name. If None, the default unique name will be returned.
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
        return namechecking.get_name(type=self.__class__.__name__)
      else:
        return namechecking.get_name(type=type)
    else:
      namechecking.check_name(name=name, obj=self)
      return name
