# -*- coding: utf-8 -*-


from brainpy.simulation.brainobjects.base import DynamicSystem
from brainpy.simulation.datastructures import DataCollector


__all__ = [
  'Function'
]


class Function(DynamicSystem):
  """Turn a function into a Module by keeping the vars it uses."""

  def __init__(self, f, all_vars):
    """Function constructor.

    Args:
        f: the function or the module to represent.
        all_vars: the Collection of variables used by the function.
    """
    if hasattr(f, '__name__'):
      self.all_vars = DataCollector((f'{{{f.__name__}}}{k}', v)
                                    for k, v in all_vars.items())
    else:
      self.all_vars = DataCollector(all_vars)
    self.__wrapped__ = f

  def __call__(self, *args, **kwargs):
    """Call the the function."""
    return self.__wrapped__(*args, **kwargs)

  def vars(self, prefix=''):
    """Return the Collection of the variables used by the function."""
    if prefix:
      return DataCollector((prefix + k, v) for k, v in self.all_vars.items())
    return DataCollector(self.all_vars)

  @staticmethod
  def with_vars(all_vars):
    """Decorator which turns a function into a module using provided variable collection.

    Parameters
    ----------
    all_vars : dict
      The Collection of variables used by the function.
    """

    def from_function(f):
      return Function(f, all_vars)

    return from_function

  def __repr__(self):
    return f'{self.__class__.__name__}(f={str(self.__wrapped__)})'

