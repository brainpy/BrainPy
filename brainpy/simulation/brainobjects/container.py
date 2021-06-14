# -*- coding: utf-8 -*-

from brainpy import errors
from .base import DynamicSystem

__all__ = [
  'Container',
  'Network',
]

_Container_NO = 0
_Network_NO = 0


class Container(DynamicSystem):
  """Container object which is designed to add other DynamicalSystem instances.

  What's different from the other DynamicSystem objects is that Container has
  one more useful function :py:func:`add`. It can be used to add the children
  objects.

  Parameters
  ----------
  steps : function, list of function, tuple of function, dict of (str, function), optional
      The step functions.
  monitors : tuple, list, Monitor, optional
      The monitor object.
  name : str, optional
      The object name.
  show_code : bool
      Whether show the formatted code.
  kwargs :
      The instance of DynamicSystem with the format of "key=value".
  """

  def __init__(self, steps=None, monitors=None, name=None, show_code=False, **kwargs):
    if name is None:
      global _Container_NO
      name = f'Container{_Container_NO}'
      _Container_NO += 1

    if monitors is not None:
      raise errors.ModelUseError(f'"monitors" cannot be used in '
                                 f'"brainpy.{self.__class__.__name__}".')

    super(Container, self).__init__(steps=steps,
                                    monitors=monitors,
                                    name=name,
                                    show_code=show_code)

    # store the step function
    self.run_func = None

    # add nodes
    self.add(**kwargs)

  def _add_obj(self, obj, name=None):
    # 1. check object type
    if not isinstance(obj, DynamicSystem):
      raise ValueError(f'Unknown object type "{type(obj)}". '
                       f'Currently, Network only supports '
                       f'"brainpy.{DynamicSystem.__name__}".')
    # 2. check object name
    name = obj.name if name is None else name
    if name in self.contained_members:
      raise KeyError(f'Name "{name}" has been used in the network, '
                     f'please change another name.')
    # 3. add object to the network
    self.contained_members[name] = obj

  def add(self, **kwargs):
    """Add object (neurons or synapses) to the network.

    Parameters
    ----------
    kwargs :
        The named objects, which can be accessed by `net.xxx`
        (xxx is the name of the object).
    """
    for name, obj in kwargs.items():
      if not isinstance(obj, DynamicSystem):
        raise ValueError(f'Unknown object type "{type(obj)}". Currently, '
                         f'{self.__class__.__name__} only supports '
                         f'"brainpy.{DynamicSystem.__name__}".')
      if hasattr(self, name):
        if not isinstance(getattr(self, name), DynamicSystem):
          raise KeyError(f'Key "{name}" has been used in this '
                         f'{self.__class__.__name__} object "{self.name}" '
                         f'to specify a "{type(getattr(self, name))}", '
                         f'please change another name.')
        else:
          print(f'WARNING: "{name}" has been used in "{self.name}", '
                f'now it is replaced by another "{DynamicSystem.__name__}" '
                f'object "{obj.name}".')
      self.contained_members[name] = obj
      setattr(self, name, obj)


class Network(Container):
  """Network object, an alias of Container.

  Network instantiates a network, which is aimed to load
  neurons, synapses, and other brain objects.

  """

  def __init__(self, name=None, **kwargs):
    if name is None:
      global _Network_NO
      name = f'Net{_Network_NO}'
      _Network_NO += 1
    super(Network, self).__init__(name=name, **kwargs)
