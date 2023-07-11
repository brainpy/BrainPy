# -*- coding: utf-8 -*-

from brainpy._src.dynsys import Dynamic
from brainpy._src.mixin import AutoDelaySupp, ParamDesc

__all__ = [
  'NeuDyn', 'SynDyn', 'IonChaDyn',
]


class NeuDyn(Dynamic, AutoDelaySupp):
  """Neuronal Dynamics."""
  pass


class SynDyn(Dynamic, AutoDelaySupp, ParamDesc):
  """Synaptic Dynamics."""
  pass


class IonChaDyn(Dynamic):
  """Ion Channel Dynamics."""
  pass

