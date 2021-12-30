# -*- coding: utf-8 -*-

from brainpy.building.brainobjects.base import DynamicalSystem

__all__ = [
  'Molecular'
]


class Molecular(DynamicalSystem):
  """Base class to model molecular objects.

  Parameters
  ----------

  steps : tuple of str, tuple of function, dict of (str, function), optional
      The callable function, or a list of callable functions.
  monitors : None, list, tuple, datastructures.Monitor
      Variables to monitor.
  name : str, optional
      The name of the dynamic system.
  """

  def __init__(self, name, **kwargs):
    super(Molecular, self).__init__(name=name, **kwargs)
