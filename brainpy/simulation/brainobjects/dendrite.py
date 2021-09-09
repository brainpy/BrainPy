# -*- coding: utf-8 -*-

from brainpy.simulation.brainobjects.base import DynamicalSystem

__all__ = [
  'Dendrite'
]


class Dendrite(DynamicalSystem):
  """Dendrite object.

  """

  def __init__(self, name, **kwargs):
    super(Dendrite, self).__init__(name=name, **kwargs)
