# -*- coding: utf-8 -*-

from typing import Union, Callable

import brainpy.math as bm
from brainpy.dyn.base import ConNeuGroup
from brainpy.initialize import Initializer, init_param
from brainpy.integrators import odeint
from brainpy.types import Shape, Tensor
from .base import IonChannel

__all__ = [
  'IhChannel',
  'Ih',
]


class IhChannel(IonChannel):
  """Base class for Ih channel models."""
  master_master_type = ConNeuGroup


class Ih(IhChannel):
  r"""The hyperpolarization-activated cation current model.

  The hyperpolarization-activated cation current model is adopted from (Huguenard, et, al., 1992) [1]_.
  Its dynamics is given by:

  .. math::

      \begin{aligned}
      I_h &= g_{\mathrm{max}} p
      \\
      \frac{dp}{dt} &= \phi \frac{p_{\infty} - p}{\tau_p}
      \\
      p_{\infty} &=\frac{1}{1+\exp ((V+75) / 5.5)}
      \\
      \tau_{p} &=\frac{1}{\exp (-0.086 V-14.59)+\exp (0.0701 V-1.87)}
      \end{aligned}

  where :math:`\phi=1` is a temperature-dependent factor.

  Parameters
  ----------
  g_max : float
    The maximal conductance density (:math:`mS/cm^2`).
  E : float
    The reversal potential (mV).
  phi : float
    The temperature-dependent factor.

  References
  ----------
  .. [1] Huguenard, John R., and David A. McCormick. "Simulation of the currents
         involved in rhythmic oscillations in thalamic relay neurons." Journal
         of neurophysiology 68, no. 4 (1992): 1373-1383.

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      g_max: Union[float, Tensor, Initializer, Callable]=10.,
      E: Union[float, Tensor, Initializer, Callable]=-90.,
      phi: Union[float, Tensor, Initializer, Callable]=1.,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(Ih, self).__init__(size, keep_size=keep_size, name=name)

    # parameters
    self.phi = init_param(phi, self.var_shape, allow_none=False)
    self.g_max = init_param(g_max, self.var_shape, allow_none=False)
    self.E = init_param(E, self.var_shape, allow_none=False)

    # variable
    self.p = bm.Variable(bm.zeros(self.var_shape))

    # function
    self.integral = odeint(self.derivative, method=method)

  def derivative(self, p, t, V):
    p_inf = 1. / (1. + bm.exp((V + 75.) / 5.5))
    p_tau = 1. / (bm.exp(-0.086 * V - 14.59) + bm.exp(0.0701 * V - 1.87))
    return self.phi * (p_inf - p) / p_tau

  def reset(self, V):
    self.p.value = 1. / (1. + bm.exp((V + 75.) / 5.5))

  def update(self, t, dt, V):
    self.p.value = self.integral(self.p, t, V, dt=dt)

  def current(self, V):
    g = self.g_max * self.p
    return g * (self.E - V)
