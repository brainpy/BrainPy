# -*- coding: utf-8 -*-

from brainpy.simulation.brainobjects.base import DynamicSystem

__all__ = [
  'Channel'
]


class Channel(DynamicSystem):
  """Ion Channel object.

  Parameters
  ----------
  name : str
      The name of the channel.

  """

  def __init__(self, name=None, **kwargs):
    super(Channel, self).__init__(name=name, **kwargs)

  def update(self, *args, **kwargs):
    pass

  def current(self, *args, **kwargs):
    pass


