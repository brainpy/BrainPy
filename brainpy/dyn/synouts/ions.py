# -*- coding: utf-8 -*-

from typing import Union, Callable

import brainpy.math as bm
from brainpy.dyn.base import SynapseOutput
from brainpy.initialize import init_param, Initializer
from brainpy.types import Tensor


__all__ = [
  'MgBlock',
]


class MgBlock(SynapseOutput):
  r"""Synaptic output based on Magnesium blocking.

  Given the synaptic conductance, the model output the post-synaptic current with

  .. math::

     I_{syn}(t) = g_{\mathrm{syn}}(t) (E - V(t)) g_{\infty}(V,[{Mg}^{2+}]_{o})

  where The fraction of channels :math:`g_{\infty}` that are not blocked by magnesium can be fitted to

  .. math::

     g_{\infty}(V,[{Mg}^{2+}]_{o}) = (1+{e}^{-\alpha V} \frac{[{Mg}^{2+}]_{o}} {\beta})^{-1}

  Here :math:`[{Mg}^{2+}]_{o}` is the extracellular magnesium concentration.

  Parameters
  ----------
  E: float, JaxArray, ndarray, callable, Initializer
    The reversal potential for the synaptic current. [mV]
  alpha: float, JaxArray, ndarray
    Binding constant. Default 0.062
  beta: float, JaxArray, ndarray, callable, Initializer
    Unbinding constant. Default 3.57
  cc_Mg: float, JaxArray, ndarray, callable, Initializer
    Concentration of Magnesium ion. Default 1.2 [mM].
  name: str
    The model name.
  """

  def __init__(
      self,
      E: Union[float, Tensor, Callable, Initializer] = 0.,
      cc_Mg: Union[float, Tensor, Callable, Initializer] = 1.2,
      alpha: Union[float, Tensor, Callable, Initializer] = 0.062,
      beta: Union[float, Tensor, Callable, Initializer] = 3.57,
      name: str = None
  ):
    super(MgBlock, self).__init__(name=name)
    self.E = E
    self.cc_Mg = cc_Mg
    self.alpha = alpha
    self.beta = beta

  def register_master(self, master):
    super(MgBlock, self).register_master(master)
    self.E = init_param(self.E, self.master.post.num, allow_none=False)
    self.cc_Mg = init_param(self.cc_Mg, self.master.post.num, allow_none=False)
    self.alpha = init_param(self.alpha, self.master.post.num, allow_none=False)
    self.beta = init_param(self.beta, self.master.post.num, allow_none=False)

  def filter(self, g):
    V = self.master.post.V.value
    return g * (self.E - V) / (1 + self.cc_Mg / self.beta * bm.exp(-self.alpha * V))

