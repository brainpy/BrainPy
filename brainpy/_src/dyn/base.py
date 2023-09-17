# -*- coding: utf-8 -*-

from brainpy._src.dynsys import Dynamic
from brainpy._src.mixin import SupportAutoDelay, ParamDesc

__all__ = [
  'NeuDyn', 'SynDyn', 'IonChaDyn',
]


class NeuDyn(Dynamic, SupportAutoDelay):
  """Neuronal Dynamics."""
  pass


class SynDyn(Dynamic, SupportAutoDelay, ParamDesc):
  """Synaptic Dynamics."""
  pass


class IonChaDyn(Dynamic):
  """Ion Channel Dynamics."""
  pass

