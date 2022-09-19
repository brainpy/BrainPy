# -*- coding: utf-8 -*-

"""
This module implements hyperpolarization-activated cation channels.

"""

from typing import Union, Callable

import brainpy.math as bm
from brainpy.initialize import Initializer, parameter, variable
from brainpy.integrators import odeint, JointEq
from brainpy.types import Shape, Array
from brainpy.modes import Mode, BatchingMode, normal
from .base import IhChannel, CalciumChannel, Calcium

__all__ = [
  'Ih_HM1992',
  'Ih_De1996',
]


class Ih_HM1992(IhChannel):
  r"""The hyperpolarization-activated cation current model propsoed by (Huguenard & McCormick, 1992) [1]_.

  The hyperpolarization-activated cation current model is adopted from
  (Huguenard, et, al., 1992) [1]_. Its dynamics is given by:

  .. math::

      \begin{aligned}
      I_h &= g_{\mathrm{max}} p \\
      \frac{dp}{dt} &= \phi \frac{p_{\infty} - p}{\tau_p} \\
      p_{\infty} &=\frac{1}{1+\exp ((V+75) / 5.5)} \\
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
      g_max: Union[float, Array, Initializer, Callable] = 10.,
      E: Union[float, Array, Initializer, Callable] = 43.,
      phi: Union[float, Array, Initializer, Callable] = 1.,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    super(Ih_HM1992, self).__init__(size,
                                    keep_size=keep_size,
                                    name=name,
                                    mode=mode)

    # parameters
    self.phi = parameter(phi, self.varshape, allow_none=False)
    self.g_max = parameter(g_max, self.varshape, allow_none=False)
    self.E = parameter(E, self.varshape, allow_none=False)

    # variable
    self.p = variable(bm.zeros, mode, self.varshape)

    # function
    self.integral = odeint(self.derivative, method=method)

  def derivative(self, p, t, V):
    return self.phi * (self.f_p_inf(V) - p) / self.f_p_tau(V)

  def reset_state(self, V, batch_size=None):
    self.p.value = self.f_p_inf(V)
    if batch_size is not None:
      assert self.p.shape[0] == batch_size

  def update(self, tdi, V):
    self.p.value = self.integral(self.p.value, tdi['t'], V, tdi['dt'])

  def current(self, V):
    return self.g_max * self.p * (self.E - V)

  def f_p_inf(self, V):
    return 1. / (1. + bm.exp((V + 75.) / 5.5))

  def f_p_tau(self, V):
    return 1. / (bm.exp(-0.086 * V - 14.59) + bm.exp(0.0701 * V - 1.87))


class Ih_De1996(IhChannel, CalciumChannel):
  r"""The hyperpolarization-activated cation current model propsoed by (Destexhe, et al., 1996) [1]_.

  The full kinetic schema was

  .. math::

     \begin{gathered}
     C \underset{\beta(V)}{\stackrel{\alpha(V)}{\rightleftarrows}} O \\
     P_{0}+2 \mathrm{Ca}^{2+} \underset{k_{2}}{\stackrel{k_{1}}{\rightleftarrows}} P_{1} \\
     O+P_{1} \underset{k_{4}}{\rightleftarrows} O_{\mathrm{L}}
     \end{gathered}

  where the first reaction represents the voltage-dependent transitions of :math:`I_h` channels
  between closed (C) and open (O) forms, with :math:`\alpha` and :math:`\beta` as transition rates.
  The second reaction represents the biding of intracellular :math:`\mathrm{Ca^{2+}}` ions to a
  regulating factor (:math:`P_0` for unbound and :math:`P_1` for bound) with four binding sites for
  calcium and rates of :math:`k_1 = 2.5e^7\, mM^{-4} \, ms^{-1}` and :math:`k_2=4e-4 \, ms^{-1}`
  (half-activation of 0.002 mM :math:`Ca^{2+}`). The calcium-bound form :math:`P_1` associates
  with the open form of the channel, leading to a locked open form :math:`O_L`, with rates of
  :math:`k_3=0.1 \, ms^{-1}` and :math:`k_4 = 0.001 \, ms^{-1}`.

  The current is the proportional to the relative concentration of open channels

  .. math::

     I_h = g_h (O+g_{inc}O_L) (V - E_h)

  with a maximal conductance of :math:`\bar{g}_{\mathrm{h}}=0.02 \mathrm{mS} / \mathrm{cm}^{2}`
  and a reversal potential of :math:`E_{\mathrm{h}}=-40 \mathrm{mV}`. Because of the factor
  :math:`g_{\text {inc }}=2`, the conductance of the calcium-bound open state of
  :math:`I_{\mathrm{h}}` channels is twice that of the unbound open state. This produces an
  augmentation of conductance after the binding of :math:`\mathrm{Ca}^{2+}`, as observed in
  sino-atrial cells (Hagiwara and Irisawa 1989).

  The rates of :math:`\alpha` and :math:`\beta` are:

  .. math::

     & \alpha = m_{\infty} / \tau_m \\
     & \beta = (1-m_{\infty}) / \tau_m \\
     & m_{\infty} = 1/(1+\exp((V+75-V_{sh})/5.5)) \\
     & \tau_m = (5.3 + 267/(\exp((V+71.5-V_{sh})/14.2) + \exp(-(V+89-V_{sh})/11.6)))

  and the temperature regulating factor :math:`\phi=2^{(T-24)/10}`.

  References
  ----------
  .. [1] Destexhe, Alain, et al. "Ionic mechanisms underlying synchronized
         oscillations and propagating waves in a model of ferret thalamic
         slices." Journal of neurophysiology 76.3 (1996): 2049-2070.
  """

  master_type = Calcium

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Array, Initializer, Callable] = -40.,
      k2: Union[float, Array, Initializer, Callable] = 4e-4,
      k4: Union[float, Array, Initializer, Callable] = 1e-3,
      V_sh: Union[float, Array, Initializer, Callable] = 0.,
      g_max: Union[float, Array, Initializer, Callable] = 0.02,
      g_inc: Union[float, Array, Initializer, Callable] = 2.,
      Ca_half: Union[float, Array, Initializer, Callable] = 2e-3,
      T: Union[float, Array] = 36.,
      T_base: Union[float, Array] = 3.,
      phi: Union[float, Array, Initializer, Callable] = None,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    # IhChannel.__init__(self, size, name=name, keep_size=keep_size)
    CalciumChannel.__init__(self,
                            size,
                            keep_size=keep_size,
                            name=name,
                            mode=mode)

    # parameters
    self.T = parameter(T, self.varshape, allow_none=False)
    self.T_base = parameter(T_base, self.varshape, allow_none=False)
    if phi is None:
      self.phi = self.T_base ** ((self.T - 24.) / 10)
    else:
      self.phi = parameter(phi, self.varshape, allow_none=False)
    self.E = parameter(E, self.varshape, allow_none=False)
    self.k2 = parameter(k2, self.varshape, allow_none=False)
    self.Ca_half = parameter(Ca_half, self.varshape, allow_none=False)
    self.k1 = self.k2 / self.Ca_half ** 4
    self.k4 = parameter(k4, self.varshape, allow_none=False)
    self.k3 = self.k4 / 0.01
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)
    self.g_max = parameter(g_max, self.varshape, allow_none=False)
    self.g_inc = parameter(g_inc, self.varshape, allow_none=False)

    # variable
    self.O = variable(bm.zeros, mode, self.varshape)
    self.OL = variable(bm.zeros, mode, self.varshape)
    self.P1 = variable(bm.zeros, mode, self.varshape)

    # function
    self.integral = odeint(JointEq(self.dO, self.dOL, self.dP1), method=method)

  def dO(self, O, t, OL, V):
    inf = self.f_inf(V)
    tau = self.f_tau(V)
    alpha = inf / tau
    beta = (1 - inf) / tau
    return alpha * (1 - O - OL) - beta * O

  def dOL(self, OL, t, O, P1):
    return self.k3 * P1 * O - self.k4 * OL

  def dP1(self, P1, t, C_Ca):
    return self.k1 * C_Ca ** 4 * (1 - P1) - self.k2 * P1

  def update(self, tdi, V, C_Ca, E_Ca):
    self.O.value = self.integral(self.O.value, self.OL.value, self.P1.value,
                                 tdi['t'], V=V, C_Ca=C_Ca, dt=tdi['dt'])

  def current(self, V, C_Ca, E_Ca):
    return self.g_max * (self.O + self.g_inc * self.OL) * (self.E - V)

  def reset_state(self, V, C_Ca, E_Ca, batch_size=None):
    varshape = self.varshape if (batch_size is None) else ((batch_size,) + self.varshape)
    self.P1.value = bm.broadcast_to(self.k1 * C_Ca ** 4 / (self.k1 * C_Ca ** 4 + self.k2), varshape)
    inf = self.f_inf(V)
    tau = self.f_tau(V)
    alpha = inf / tau
    beta = (1 - inf) / tau
    self.O.value = alpha / (alpha + alpha * self.k3 * self.P1 / self.k4 + beta)
    self.OL.value = self.k3 * self.P1 * self.O / self.k4
    if batch_size is not None:
      assert self.P1.shape[0] == batch_size
      assert self.O.shape[0] == batch_size
      assert self.OL.shape[0] == batch_size

  def f_inf(self, V):
    return 1 / (1 + bm.exp((V + 75 - self.V_sh) / 5.5))

  def f_tau(self, V):
    return (20. + 1000 / (bm.exp((V + 71.5 - self.V_sh) / 14.2) +
                          bm.exp(-(V + 89 - self.V_sh) / 11.6))) / self.phi
