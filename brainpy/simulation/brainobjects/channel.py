# -*- coding: utf-8 -*-

from .base import DynamicSystem

__all__ = [
  'Channel'
]

_Channel_NO = 0


class Channel(DynamicSystem):
  """Ion Channel object.

  Parameters
  ----------
  name : str
      The name of the channel.

  """

  def __init__(self, name=None, **kwargs):
    if name is None:
      global _Channel_NO
      name = f'Channel{_Channel_NO}'
      _Channel_NO += 1

    super(Channel, self).__init__(name=name, **kwargs)
