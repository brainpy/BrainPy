# -*- coding: utf-8 -*-

"""
This module implements leakage channels.

"""

from typing import Union, Callable, Sequence

import brainpy.math as bm
from brainpy._src.dyn.neurons.hh import HHTypedNeuron
from brainpy._src.initialize import Initializer, parameter
from brainpy.types import ArrayType
from .base import IonChannel

__all__ = [
  'LeakyChannel',
  'IL',
]


class LeakyChannel(IonChannel):
  """Base class for leaky channel dynamics."""

  master_type = HHTypedNeuron

  def reset_state(self, V, batch_size=None):
    pass


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
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      g_max: Union[int, float, ArrayType, Initializer, Callable] = 0.1,
      E: Union[int, float, ArrayType, Initializer, Callable] = -70.,
      method: str = None,
      name: str = None,
      mode: bm.Mode = None,
  ):
    super(IL, self).__init__(size,
                             keep_size=keep_size,
                             name=name,
                             mode=mode)

    self.E = parameter(E, self.varshape, allow_none=False)
    self.g_max = parameter(g_max, self.varshape, allow_none=False)
    self.method = method

  def reset_state(self, V, batch_size=None):
    pass

  def update(self, V):
    pass

  def current(self, V):
    return self.g_max * (self.E - V)
