# -*- coding: utf-8 -*-

from brainpy.types import Shape

from brainpy.initialize import init_param
from brainpy.dyn.base import ConNeuGroup
from .base import IonChannel

__all__ = [
  'LeakyChannel',
  'IL',
  'IKL',
]


class LeakyChannel(IonChannel):
  """Base class for leaky channel."""
  master_master_type = ConNeuGroup


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
      keep_size: bool = False,
      g_max=0.1,
      E=-70.,
      method: str = None,
      name: str = None,
  ):
    super(IL, self).__init__(size, keep_size=keep_size, name=name)

    self.E = init_param(E, self.var_shape, allow_none=False)
    self.g_max = init_param(g_max, self.var_shape, allow_none=False)
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
      keep_size: bool = False,
      g_max=0.005,
      E=-90.,
      method=None,
      name=None,
  ):
    super(IKL, self).__init__(size=size, keep_size=keep_size, g_max=g_max, E=E, method=method, name=name)
