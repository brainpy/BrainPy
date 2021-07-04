# -*- coding: utf-8 -*-

from brainpy.simulation.brainobjects.base import DynamicSystem

__all__ = [
  'Molecular'
]


class Molecular(DynamicSystem):
  """Molecular object for neuron modeling.

  """

  def __init__(self, name, **kwargs):
    super(Molecular, self).__init__(name=self.unique_name(name, 'Molecular'),
                                    **kwargs)
