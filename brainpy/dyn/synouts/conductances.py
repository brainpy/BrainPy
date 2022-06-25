# -*- coding: utf-8 -*-

from typing import Union, Callable

from brainpy.dyn.base import SynOutput
from brainpy.initialize import parameter, Initializer
from brainpy.types import Tensor

__all__ = [
  'COBA',
  'CUBA',
]


class CUBA(SynOutput):
  r"""Current-based synaptic output.

  Given the conductance, this model outputs the post-synaptic current with a identity function:

  .. math::

     I_{\mathrm{syn}}(t) = g_{\mathrm{syn}}(t)

  Parameters
  ----------
  name: str
    The model name.


  See Also
  --------
  COBA
  """

  def __init__(self, name: str = None):
    super(CUBA, self).__init__(name=name)

  def update(self, tdi):
    pass

  def filter(self, g):
    return g


class COBA(SynOutput):
  r"""Conductance-based synaptic output.

  Given the synaptic conductance, the model output the post-synaptic current with

  .. math::

     I_{syn}(t) = g_{\mathrm{syn}}(t) (E - V(t))

  Parameters
  ----------
  E: float, JaxArray, ndarray, callable, Initializer
    The reversal potential.
  name: str
    The model name.

  See Also
  --------
  CUBA
  """

  def __init__(
      self,
      E: Union[float, Tensor, Callable, Initializer] = 0.,
      name: str = None
  ):
    super(COBA, self).__init__(name=name)
    self._E = E

  def register_master(self, master):
    super(COBA, self).register_master(master)
    self.E = parameter(self._E, self.master.post.num, allow_none=False)

  def filter(self, g):
    return g * (self.E - self.master.post.V)

  def update(self, tdi):
    pass
