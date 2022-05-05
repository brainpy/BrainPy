# -*- coding: utf-8 -*-

from brainpy.types import Shape

from brainpy.dyn.base import ConNeuGroup
from .base import IonChannel

__all__ = [
  'LeakyChannel',
  'IL',
  'IKL',
]


class LeakyChannel(IonChannel):
  """Base class for leaky channel."""
  master_cls = ConNeuGroup


class IL(LeakyChannel):
  """The leakage channel current.

  Parameters
  ----------
  g_max : float
    The leakage conductance.
  E : float
    The reversal potential.
  """

  def __init__(
      self,
      size,
      g_max=0.1,
      E=-70.,
      method: str = None,
      name: str = None,
  ):
    super(IL, self).__init__(size, name=name)

    self.E = E
    self.g_max = g_max
    self.method = method

  def reset(self, V):
    pass

  def update(self, t, dt, V):
    pass

  def current(self, V):
    return self.g_max * (self.E - V)


class IKL(IL):
  """The potassium leak channel current.

  Parameters
  ----------
  g_max : float
    The potassium leakage conductance which is modulated by both
    acetylcholine and norepinephrine.
  E : float
    The reversal potential.
  """

  def __init__(
      self,
      size: Shape,
      g_max=0.005,
      E=-90.,
      method=None,
      name=None,
  ):
    super(IKL, self).__init__(size=size, g_max=g_max, E=E, method=method, name=name)
