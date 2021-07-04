# -*- coding: utf-8 -*-

from brainpy.simulation.brainobjects.base import DynamicSystem

__all__ = [
  'Dendrite'
]


class Dendrite(DynamicSystem):
  """Dendrite object.

  """

  def __init__(self, name, **kwargs):
    super(Dendrite, self).__init__(name=self.unique_name(name, 'Dendrite'), **kwargs)
