# -*- coding: utf-8 -*-


from brainpy import math
from brainpy.base import collector
from brainpy.tools import namechecking
from brainpy.integrators import constants

__all__ = [
  'Base',
]


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
    gather = collector.ArrayCollector()
    if method == 'absolute':
      for k, v in self.__dict__.items():
        if isinstance(v, math.ndarray):
          gather[f'{self.name}.{k}'] = v
        elif isinstance(v, Base):
          gather.update(v.vars(method=method))
    elif method == 'relative':
      for k, v in self.__dict__.items():
        if isinstance(v, math.ndarray):
          gather[k] = v
        elif isinstance(v, Base):
          for k2, v2 in v.vars(method=method).items():
            gather[f'{k}.{k2}'] = v2
    else:
      raise ValueError(f'No support for the method of "{method}".')
    return gather

  def train_vars(self, method='absolute'):
    return self.vars(method=method)

  def nodes(self, method='absolute'):
    """Collect all children nodes.

    Parameters
    ----------
    method : str
      The method to access the nodes.

    Returns
    -------
    gather : collector.Collector
      The collection contained (the path, the node).
    """
    gather = collector.Collector()
    if method == 'absolute':
      for k, v in self.__dict__.items():
        if isinstance(v, Base):
          gather[v.name] = v
          gather.update(v.nodes(method=method))
    elif method == 'relative':
      for k, v in self.__dict__.items():
        if isinstance(v, Base):
          gather[k] = v
          for k2, v2 in v.nodes(method=method).items():
            gather[f'{k}.{k2}'] = v2
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
    gather = collector.Collector()
    if method == 'absolute':
      for k in dir(self):
        v = getattr(self, k)
        if callable(v) and hasattr(v, '__name__') and v.__name__.startswith(constants.DE_INT):
          gather[f'{self.name}.{k}'] = v
        elif isinstance(v, Base):
          gather.update(v.ints(method=method))
    elif method == 'relative':
      for k in dir(self):
        v = getattr(self, k)
        if callable(v) and hasattr(v, '__name__') and v.__name__.startswith(constants.DE_INT):
          gather[k] = v
        elif isinstance(v, Base):
          for k2, v2 in v.ints(method=method).items():
            gather[f'{k}.{k2}'] = v2
    else:
      raise ValueError(f'No support for the method of "{method}".')
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

