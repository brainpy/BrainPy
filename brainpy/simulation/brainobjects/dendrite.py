# -*- coding: utf-8 -*-

from .base import DynamicSystem

__all__ = [
  'Dendrite'
]

_Dendrite_NO = 0


class Dendrite(DynamicSystem):
  """Dendrite object.

  """

  def __init__(self, name, **kwargs):
    if name is None:
      global _Dendrite_NO
      name = f'Dendrite{_Dendrite_NO}'
      _Dendrite_NO += 1
    super(Dendrite, self).__init__(name=name, **kwargs)
