# -*- coding: utf-8 -*-

from brainpy.simulation.brainobjects.base import DynamicalSystem

__all__ = [
  'Soma'
]


class Soma(DynamicalSystem):
  """Soma object for neuron modeling.

  """

  def __init__(self, name, **kwargs):
    super(Soma, self).__init__(name=name, **kwargs)
