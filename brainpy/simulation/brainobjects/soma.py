# -*- coding: utf-8 -*-

from brainpy.simulation.brainobjects.base import DynamicSystem

__all__ = [
  'Soma'
]


class Soma(DynamicSystem):
  """Soma object for neuron modeling.

  """

  def __init__(self, name, **kwargs):
    super(Soma, self).__init__(name=self.unique_name(name, 'Soma'),
                               **kwargs)
