# -*- coding: utf-8 -*-

from brainpy.simulation.brainobjects.base import DynamicalSystem

__all__ = [
  'Molecular'
]


class Molecular(DynamicalSystem):
  """Molecular object for neuron modeling.

  """

  def __init__(self, name, **kwargs):
    super(Molecular, self).__init__(name=name, **kwargs)
