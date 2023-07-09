# -*- coding: utf-8 -*-

from brainpy._src.dynsys import Dynamics
from brainpy._src.mixin import AutoDelaySupp, ParamDesc

__all__ = [
  'NeuDyn', 'SynDyn', 'IonChaDyn',
]


class NeuDyn(Dynamics, AutoDelaySupp):
  """Neuronal Dynamics."""
  pass


class SynDyn(Dynamics, AutoDelaySupp, ParamDesc):
  """Synaptic Dynamics."""
  pass


class IonChaDyn(Dynamics):
  """Ion Channel Dynamics."""
  pass

