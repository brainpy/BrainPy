# -*- coding: utf-8 -*-

from brainpy import errors, math
from brainpy.integrators.integrators import Integrator
from brainpy.simulation.brainobjects.base import DynamicSystem
from brainpy.tools.collector import Collector

__all__ = [
  'Container',
]

_Container_NO = 0


class Container(DynamicSystem, list):
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

  def vars(self, prefix=''):
    """Collect all the variables (and their names) contained
    in the list and its children instance of DynamicSystem.

    Parameters
    ----------
    prefix : str
      string to prefix to the variable names.

    Returns
    -------
    collection : Collector
        A DataCollector of all the variables.
    """
    collector = Collector()
    prefix1 = prefix + f'({self.name})'
    for i, v in enumerate(self):
      if isinstance(v, math.ndarray):
        collector[f'{prefix1}[{i}]'] = v
      elif isinstance(v, DynamicSystem):
        collector.update(v.vars(prefix=f'{prefix1}[{i}]'))
    prefix2 = prefix + f'({self.name}).'
    for k, v in self.__dict__.items():
      if isinstance(v, math.ndarray):
        collector[prefix2 + k] = v
      elif isinstance(v, DynamicSystem):
        collector.update(v.vars(prefix=prefix2[:-1] if k == 'raw' else prefix2 + k))
    return collector

  def ints(self, prefix=''):
    collector = Collector()
    prefix1 = prefix + f'({self.name})'
    for i, v in enumerate(self):
      if isinstance(v, Integrator):
        collector[f'{prefix1}[{i}]'] = v
      elif isinstance(v, DynamicSystem):
        collector.update(v.ints(prefix=f'{prefix1}[{i}]'))
    prefix2 = prefix + f'({self.name}).'
    for k, v in self.__dict__.items():
      if isinstance(v, Integrator):
        collector[prefix2 + k] = v
      elif isinstance(v, DynamicSystem):
        collector.update(v.ints(prefix=prefix2[:-1] if k == 'raw' else prefix2 + k))
    return collector

  def nodes(self, prefix=''):
    collector = Collector()
    prefix += f'{self.name}.'
    for v in self:
      collector[v.name] = v
      collector.update(v.nodes(prefix[:-1]))
    for k, v in self.__dict__.items():
      if isinstance(v, DynamicSystem):
        collector[v.name] = v
        collector[prefix + f'{k}'] = v
        collector.update(v.nodes(prefix + f'{k}.'))
    return collector

  def __getitem__(self, key):
    """Get the item by slice.

    Parameters
    ----------
    key : int, slice

    Returns
    -------
    dynamic_system : DynamicSystem
      The selected children items.
    """
    value = list.__getitem__(self, key)
    if isinstance(key, slice):
      return type(self.__class__)(value)
    return value

  def __init__(self, *args, steps=None, monitors=None, name=None, **kwargs):
    if name is None:
      global _Container_NO
      name = f'Container{_Container_NO}'
      _Container_NO += 1

    if monitors is not None:
      raise errors.ModelUseError(f'"monitors" cannot be used in '
                                 f'"brainpy.{self.__class__.__name__}".')

    DynamicSystem.__init__(self, steps=steps, monitors=monitors, name=name)
    for arg in args:
      if not isinstance(arg, DynamicSystem):
        raise errors.ModelUseError(f'{self.__class__.__name__} receives '
                                   f'instances of DynamicSystem, however, '
                                   f'we got {type(arg)}.')
    list.__init__(self, args)
