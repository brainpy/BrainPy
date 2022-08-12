# -*- coding: utf-8 -*-

"""
This module implements voltage-dependent calcium channels.

"""

from typing import Union, Callable

import brainpy.math as bm
from brainpy.dyn.base import Channel
from brainpy.initialize import OneInit, Initializer, parameter, variable
from brainpy.integrators.joint_eq import JointEq
from brainpy.integrators.ode import odeint
from brainpy.types import Shape, Array
from brainpy.modes import Mode, BatchingMode, normal
from .base import Calcium, CalciumChannel

__all__ = [
  'CalciumFixed',
  'CalciumDyna',
  'CalciumDetailed',
  'CalciumFirstOrder',

  'ICa_p2q_ss', 'ICa_p2q_markov',

  'ICaN_IS2008',

  'ICaT_HM1992',
  'ICaT_HP1992',

  'ICaHT_HM1992',

  'ICaL_IS2008',
]


class CalciumFixed(Calcium):
  """Fixed Calcium dynamics.

  This calcium model has no dynamics. It holds fixed reversal
  potential :math:`E` and concentration :math:`C`.
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Array, Initializer, Callable] = 120.,
      C: Union[float, Array, Initializer, Callable] = 2.4e-4,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
      **channels
  ):
    super(CalciumFixed, self).__init__(size,
                                       keep_size=keep_size,
                                       method=method,
                                       name=name,
                                       mode=mode,
                                       **channels)
    self.E = parameter(E, self.varshape, allow_none=False)
    self.C = parameter(C, self.varshape, allow_none=False)

  def update(self, tdi, V):
    for node in self.implicit_nodes.values():
      node.update(tdi, V, self.C, self.E)

  def reset_state(self, V, C_Ca=None, E_Ca=None, batch_size=None):
    C_Ca = self.C if C_Ca is None else C_Ca
    E_Ca = self.E if E_Ca is None else E_Ca
    for node in self.nodes(level=1, include_self=False).unique().subset(Channel).values():
      node.reset_state(V, C_Ca, E_Ca, batch_size=batch_size)


class CalciumDyna(Calcium):
  """Calcium ion flow with dynamics.

  Parameters
  ----------
  size: int, tuple of int
    The ion size.
  keep_size: bool
    Keep the geometry size.
  C0: float, Array, Initializer, Callable
    The Calcium concentration outside of membrane.
  T: float, Array, Initializer, Callable
    The temperature.
  C_initializer: Initializer, Callable, Array
    The initializer for Calcium concentration.
  method: str
    The numerical method.
  name: str
    The ion name.
  """
  R = 8.31441  # gas constant, J*mol-1*K-1
  F = 96.489  # the Faraday constant

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      C0: Union[float, Array, Initializer, Callable] = 2.,
      T: Union[float, Array, Initializer, Callable] = 36.,
      C_initializer: Union[Initializer, Callable, Array] = OneInit(2.4e-4),
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
      **channels
  ):
    super(CalciumDyna, self).__init__(size,
                                      keep_size=keep_size,
                                      method=method,
                                      name=name,
                                      mode=mode,
                                      **channels)

    # parameters
    self.C0 = parameter(C0, self.varshape, allow_none=False)
    self.T = parameter(T, self.varshape, allow_none=False)  # temperature
    self._C_initializer = C_initializer
    self._constant = self.R / (2 * self.F) * (273.15 + self.T)

    # variables
    self.C = variable(C_initializer, mode, self.varshape)  # Calcium concentration
    self.E = bm.Variable(self._reversal_potential(self.C),
                         batch_axis=0 if isinstance(mode, BatchingMode) else None)  # Reversal potential

    # function
    self.integral = odeint(self.derivative, method=method)

  def derivative(self, C, t, V):
    raise NotImplementedError

  def reset_state(self, V, C_Ca=None, E_Ca=None, batch_size=None):
    self.C.value = variable(self._C_initializer, batch_size, self.varshape) if (C_Ca is None) else C_Ca
    self.E.value = self._reversal_potential(self.C)
    for node in self.nodes(level=1, include_self=False).unique().subset(Channel).values():
      node.reset(V, self.C, self.E, batch_size=batch_size)

  def update(self, tdi, V):
    for node in self.nodes(level=1, include_self=False).unique().subset(Channel).values():
      node.update(tdi, V, self.C, self.E)
    self.C.value = self.integral(self.C.value, tdi['t'], V, tdi['dt'])
    self.E.value = self._reversal_potential(self.C)

  def _reversal_potential(self, C):
    return self._constant * bm.log(self.C0 / C)


class CalciumDetailed(CalciumDyna):
  r"""Dynamical Calcium model proposed.

  **1. The dynamics of intracellular** :math:`Ca^{2+}`

  The dynamics of intracellular :math:`Ca^{2+}` were determined by two contributions [1]_ :

  *(i) Influx of* :math:`Ca^{2+}` *due to Calcium currents*

  :math:`Ca^{2+}` ions enter through :math:`Ca^{2+}` channels and diffuse into the
  interior of the cell. Only the :math:`Ca^{2+}` concentration in a thin shell beneath
  the membrane was modeled. The influx of :math:`Ca^{2+}` into such a thin shell followed:

  .. math::

      [Ca]_{i}=-\frac{k}{2 F d} I_{Ca}

  where :math:`F=96489\, \mathrm{C\, mol^{-1}}` is the Faraday constant,
  :math:`d=1\, \mathrm{\mu m}` is the depth of the shell beneath the membrane,
  the unit conversion constant is :math:`k=0.1` for :math:`I_T` in
  :math:`\mathrm{\mu A/cm^{2}}` and :math:`[Ca]_{i}` in millimolar,
  and :math:`I_{Ca}` is the summation of all :math:`Ca^{2+}` currents.

  *(ii) Efflux of* :math:`Ca^{2+}` *due to an active pump*

  In a thin shell beneath the membrane, :math:`Ca^{2+}` retrieval usually consists of a
  combination of several processes, such as binding to :math:`Ca^{2+}` buffers, calcium
  efflux due to :math:`Ca^{2+}` ATPase pump activity and diffusion to neighboring shells.
  Only the :math:`Ca^{2+}` pump was modeled here. We adopted the following kinetic scheme:

  .. math::

      Ca _{i}^{2+}+ P \overset{c_1}{\underset{c_2}{\rightleftharpoons}} CaP \xrightarrow{c_3} P+ Ca _{0}^{2+}

  where P represents the :math:`Ca^{2+}` pump, CaP is an intermediate state,
  :math:`Ca _{ o }^{2+}` is the extracellular :math:`Ca^{2+}` concentration,
  and :math:`c_{1}, c_{2}` and :math:`c_{3}` are rate constants. :math:`Ca^{2+}`
  ions have a high affinity for the pump :math:`P`, whereas extrusion of
  :math:`Ca^{2+}` follows a slower process (Blaustein, 1988 ). Therefore,
  :math:`c_{3}` is low compared to :math:`c_{1}` and :math:`c_{2}` and the
  Michaelis-Menten approximation can be used for describing the kinetics of the pump.
  According to such a scheme, the kinetic equation for the :math:`Ca^{2+}` pump is:

  .. math::

      \frac{[Ca^{2+}]_{i}}{dt}=-\frac{K_{T}[Ca]_{i}}{[Ca]_{i}+K_{d}}

  where :math:`K_{T}=10^{-4}\, \mathrm{mM\, ms^{-1}}` is the product of :math:`c_{3}`
  with the total concentration of :math:`P` and :math:`K_{d}=c_{2} / c_{1}=10^{-4}\, \mathrm{mM}`
  is the dissociation constant, which can be interpreted here as the value of
  :math:`[Ca]_{i}` at which the pump is half activated (if :math:`[Ca]_{i} \ll K_{d}`
  then the efflux is negligible).

  **2.A simple first-order model**

  While, in (Bazhenov, et al., 1998) [2]_, the :math:`Ca^{2+}` dynamics is
  described by a simple first-order model,

  .. math::

      \frac{d\left[Ca^{2+}\right]_{i}}{d t}=-\frac{I_{Ca}}{z F d}+\frac{\left[Ca^{2+}\right]_{rest}-\left[C a^{2+}\right]_{i}}{\tau_{Ca}}

  where :math:`I_{Ca}` is the summation of all :math:`Ca ^{2+}` currents, :math:`d`
  is the thickness of the perimembrane "shell" in which calcium is able to affect
  membrane properties :math:`(1.\, \mathrm{\mu M})`, :math:`z=2` is the valence of the
  :math:`Ca ^{2+}` ion, :math:`F` is the Faraday constant, and :math:`\tau_{C a}` is
  the :math:`Ca ^{2+}` removal rate. The resting :math:`Ca ^{2+}` concentration was
  set to be :math:`\left[ Ca ^{2+}\right]_{\text {rest}}=.05\, \mathrm{\mu M}` .

  **3. The reversal potential**

  The reversal potential of calcium :math:`Ca ^{2+}` is calculated according to the
  Nernst equation:

  .. math::

      E = k'{RT \over 2F} log{[Ca^{2+}]_0 \over [Ca^{2+}]_i}

  where :math:`R=8.31441 \, \mathrm{J} /(\mathrm{mol}^{\circ} \mathrm{K})`,
  :math:`T=309.15^{\circ} \mathrm{K}`,
  :math:`F=96,489 \mathrm{C} / \mathrm{mol}`,
  and :math:`\left[\mathrm{Ca}^{2+}\right]_{0}=2 \mathrm{mM}`.

  Parameters
  ----------
  d : float
    The thickness of the peri-membrane "shell".
  F : float
    The Faraday constant. (:math:`C*mmol^{-1}`)
  tau : float
    The time constant of the :math:`Ca ^{2+}` removal rate. (ms)
  C_rest : float
    The resting :math:`Ca ^{2+}` concentration.
  C0 : float
    The :math:`Ca ^{2+}` concentration outside of the membrane.
  R : float
    The gas constant. (:math:` J*mol^{-1}*K^{-1}`)

  References
  ----------

  .. [1] Destexhe, Alain, Agnessa Babloyantz, and Terrence J. Sejnowski.
         "Ionic mechanisms for intrinsic slow oscillations in thalamic
         relay neurons." Biophysical journal 65, no. 4 (1993): 1538-1552.
  .. [2] Bazhenov, Maxim, Igor Timofeev, Mircea Steriade, and Terrence J.
         Sejnowski. "Cellular and network models for intrathalamic augmenting
         responses during 10-Hz stimulation." Journal of neurophysiology 79,
         no. 5 (1998): 2730-2748.

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      T: Union[float, Array, Initializer, Callable] = 36.,
      d: Union[float, Array, Initializer, Callable] = 1.,
      C_rest: Union[float, Array, Initializer, Callable] = 2.4e-4,
      tau: Union[float, Array, Initializer, Callable] = 5.,
      C0: Union[float, Array, Initializer, Callable] = 2.,
      C_initializer: Union[Initializer, Callable, Array] = OneInit(2.4e-4),
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
      **channels
  ):
    super(CalciumDetailed, self).__init__(size,
                                          keep_size=keep_size,
                                          method=method,
                                          name=name,
                                          T=T,
                                          C0=C0,
                                          C_initializer=C_initializer,
                                          mode=mode,
                                          **channels)

    # parameters
    self.d = parameter(d, self.varshape, allow_none=False)
    self.tau = parameter(tau, self.varshape, allow_none=False)
    self.C_rest = parameter(C_rest, self.varshape, allow_none=False)

  def derivative(self, C, t, V):
    ICa = self.current(V, C, self.E)
    drive = bm.maximum(- ICa / (2 * self.F * self.d), 0.)
    return drive + (self.C_rest - C) / self.tau


class CalciumFirstOrder(CalciumDyna):
  r"""The first-order calcium concentration model.

  .. math::

     Ca' = -\alpha I_{Ca} + -\beta Ca

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      T: Union[float, Array, Initializer, Callable] = 36.,
      alpha: Union[float, Array, Initializer, Callable] = 0.13,
      beta: Union[float, Array, Initializer, Callable] = 0.075,
      C0: Union[float, Array, Initializer, Callable] = 2.,
      C_initializer: Union[Initializer, Callable, Array] = OneInit(2.4e-4),
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
      **channels
  ):
    super(CalciumFirstOrder, self).__init__(size,
                                            keep_size=keep_size,
                                            method=method,
                                            name=name,
                                            T=T,
                                            C0=C0,
                                            C_initializer=C_initializer,
                                            mode=mode,
                                            **channels)

    # parameters
    self.alpha = parameter(alpha, self.varshape, allow_none=False)
    self.beta = parameter(beta, self.varshape, allow_none=False)

  def derivative(self, C, t, V):
    ICa = self.current(V, C, self.E)
    drive = bm.maximum(- self.alpha * ICa, 0.)
    return drive - self.beta * C


# -------------------------


class ICa_p2q_ss(CalciumChannel):
  r"""The calcium current model of :math:`p^2q` current which described with steady-state format.

  The dynamics of this generalized calcium current model is given by:

  .. math::

      I_{CaT} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= {\phi_p \cdot (p_{\infty}-p)\over \tau_p} \\
      {dq \over dt} &= {\phi_q \cdot (q_{\infty}-q) \over \tau_q} \\

  where :math:`\phi_p` and :math:`\phi_q` are temperature-dependent factors,
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  size: int, tuple of int
    The size of the simulation target.
  keep_size: bool
    Keep size or flatten the size?
  method: str
    The numerical method
  name: str
    The name of the object.
  g_max : float, Array, Callable, Initializer
    The maximum conductance.
  phi_p : float, Array, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : float, Array, Callable, Initializer
    The temperature factor for channel :math:`q`.

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      phi_p: Union[float, Array, Initializer, Callable] = 3.,
      phi_q: Union[float, Array, Initializer, Callable] = 3.,
      g_max: Union[float, Array, Initializer, Callable] = 2.,
      method: str = 'exp_auto',
      mode: Mode = normal,
      name: str = None
  ):
    super(ICa_p2q_ss, self).__init__(size,
                                     keep_size=keep_size,
                                     name=name,
                                     mode=mode, )

    # parameters
    self.phi_p = parameter(phi_p, self.varshape, allow_none=False)
    self.phi_q = parameter(phi_q, self.varshape, allow_none=False)
    self.g_max = parameter(g_max, self.varshape, allow_none=False)

    # variables
    self.p = variable(bm.zeros, mode, self.varshape)
    self.q = variable(bm.zeros, mode, self.varshape)

    # functions
    self.integral = odeint(JointEq([self.dp, self.dq]), method=method)

  def dp(self, p, t, V):
    return self.phi_p * (self.f_p_inf(V) - p) / self.f_p_tau(V)

  def dq(self, q, t, V):
    return self.phi_q * (self.f_q_inf(V) - q) / self.f_q_tau(V)

  def update(self, tdi, V, C_Ca, E_Ca):
    self.p.value, self.q.value = self.integral(self.p, self.q, tdi['t'], V, tdi['dt'])

  def current(self, V, C_Ca, E_Ca):
    return self.g_max * self.p * self.p * self.q * (E_Ca - V)

  def reset_state(self, V, C_Ca, E_Ca, batch_size=None):
    self.p.value = self.f_p_inf(V)
    self.q.value = self.f_q_inf(V)
    if batch_size is not None:
      assert self.p.shape[0] == batch_size
      assert self.q.shape[0] == batch_size

  def f_p_inf(self, V):
    raise NotImplementedError

  def f_p_tau(self, V):
    raise NotImplementedError

  def f_q_inf(self, V):
    raise NotImplementedError

  def f_q_tau(self, V):
    raise NotImplementedError


class ICa_p2q_markov(CalciumChannel):
  r"""The calcium current model of :math:`p^2q` current which described with first-order Markov chain.

  The dynamics of this generalized calcium current model is given by:

  .. math::

      I_{CaT} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= \phi_p (\alpha_p(V)(1-p) - \beta_p(V)p) \\
      {dq \over dt} &= \phi_q (\alpha_q(V)(1-q) - \beta_q(V)q) \\

  where :math:`\phi_p` and :math:`\phi_q` are temperature-dependent factors,
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  size: int, tuple of int
    The size of the simulation target.
  keep_size: bool
    Keep size or flatten the size?
  method: str
    The numerical method
  name: str
    The name of the object.
  g_max : float, Array, Callable, Initializer
    The maximum conductance.
  phi_p : float, Array, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : float, Array, Callable, Initializer
    The temperature factor for channel :math:`q`.

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      phi_p: Union[float, Array, Initializer, Callable] = 3.,
      phi_q: Union[float, Array, Initializer, Callable] = 3.,
      g_max: Union[float, Array, Initializer, Callable] = 2.,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    super(ICa_p2q_markov, self).__init__(size,
                                         keep_size=keep_size,
                                         name=name,
                                         mode=mode)

    # parameters
    self.phi_p = parameter(phi_p, self.varshape, allow_none=False)
    self.phi_q = parameter(phi_q, self.varshape, allow_none=False)
    self.g_max = parameter(g_max, self.varshape, allow_none=False)

    # variables
    self.p = variable(bm.zeros, mode, self.varshape)
    self.q = variable(bm.zeros, mode, self.varshape)

    # functions
    self.integral = odeint(JointEq([self.dp, self.dq]), method=method)

  def dp(self, p, t, V):
    return self.phi_p * (self.f_p_alpha(V) * (1 - p) - self.f_p_beta(V) * p)

  def dq(self, q, t, V):
    return self.phi_q * (self.f_q_alpha(V) * (1 - q) - self.f_q_beta(V) * q)

  def update(self, tdi, V, C_Ca, E_Ca):
    self.p.value, self.q.value = self.integral(self.p, self.q, tdi['t'], V, tdi['dt'])

  def current(self, V, C_Ca, E_Ca):
    return self.g_max * self.p * self.p * self.q * (E_Ca - V)

  def reset_state(self, V, C_Ca, E_Ca, batch_size=None):
    alpha, beta = self.f_p_alpha(V), self.f_p_beta(V)
    self.p.value = alpha / (alpha + beta)
    alpha, beta = self.f_q_alpha(V), self.f_q_beta(V)
    self.q.value = alpha / (alpha + beta)
    if batch_size is not None:
      assert self.p.shape[0] == batch_size
      assert self.q.shape[0] == batch_size

  def f_p_alpha(self, V):
    raise NotImplementedError

  def f_p_beta(self, V):
    raise NotImplementedError

  def f_q_alpha(self, V):
    raise NotImplementedError

  def f_q_beta(self, V):
    raise NotImplementedError


class ICaN_IS2008(CalciumChannel):
  r"""The calcium-activated non-selective cation channel model
  proposed by (Inoue & Strowbridge, 2008) [2]_.

  The dynamics of the calcium-activated non-selective cation channel model [1]_ [2]_ is given by:

  .. math::

      \begin{aligned}
      I_{CAN} &=g_{\mathrm{max}} M\left([Ca^{2+}]_{i}\right) p \left(V-E\right)\\
      &M\left([Ca^{2+}]_{i}\right) ={[Ca^{2+}]_{i} \over 0.2+[Ca^{2+}]_{i}} \\
      &{dp \over dt} = {\phi \cdot (p_{\infty}-p)\over \tau_p} \\
      &p_{\infty} = {1.0 \over 1 + \exp(-(V + 43) / 5.2)} \\
      &\tau_{p} = {2.7 \over \exp(-(V + 55) / 15) + \exp((V + 55) / 15)} + 1.6
      \end{aligned}

  where :math:`\phi` is the temperature factor.

  Parameters
  ----------
  g_max : float
    The maximal conductance density (:math:`mS/cm^2`).
  E : float
    The reversal potential (mV).
  phi : float
    The temperature factor.

  References
  ----------

  .. [1] Destexhe, Alain, et al. "A model of spindle rhythmicity in the isolated
         thalamic reticular nucleus." Journal of neurophysiology 72.2 (1994): 803-818.
  .. [2] Inoue T, Strowbridge BW (2008) Transient activity induces a long-lasting
         increase in the excitability of olfactory bulb interneurons.
         J Neurophysiol 99: 187–199.
  """

  '''The type of the master object.'''
  master_type = CalciumDyna

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Array, Initializer, Callable] = 10.,
      g_max: Union[float, Array, Initializer, Callable] = 1.,
      phi: Union[float, Array, Initializer, Callable] = 1.,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    super(ICaN_IS2008, self).__init__(size,
                                      keep_size=keep_size,
                                      name=name,
                                      mode=mode)

    # parameters
    self.E = parameter(E, self.varshape, allow_none=False)
    self.g_max = parameter(g_max, self.varshape, allow_none=False)
    self.phi = parameter(phi, self.varshape, allow_none=False)

    # variables
    self.p = variable(bm.zeros, mode, self.varshape)

    # function
    self.integral = odeint(self.derivative, method=method)

  def derivative(self, p, t, V):
    phi_p = 1.0 / (1 + bm.exp(-(V + 43.) / 5.2))
    p_inf = 2.7 / (bm.exp(-(V + 55.) / 15.) + bm.exp((V + 55.) / 15.)) + 1.6
    return self.phi * (phi_p - p) / p_inf

  def update(self, tdi, V, C_Ca, E_Ca):
    self.p.value = self.integral(self.p, tdi['t'], V, tdi['dt'])

  def current(self, V, C_Ca, E_Ca):
    M = C_Ca / (C_Ca + 0.2)
    g = self.g_max * M * self.p
    return g * (self.E - V)

  def reset_state(self, V, C_Ca, E_Ca, batch_size=None):
    self.p.value = 1.0 / (1 + bm.exp(-(V + 43.) / 5.2))
    if batch_size is not None:
      assert self.p.shape[0] == batch_size


class ICaT_HM1992(ICa_p2q_ss):
  r"""The low-threshold T-type calcium current model proposed by (Huguenard & McCormick, 1992) [1]_.

  The dynamics of the low-threshold T-type calcium current model [1]_ is given by:

  .. math::

      I_{CaT} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= {\phi_p \cdot (p_{\infty}-p)\over \tau_p} \\
      &p_{\infty} = {1 \over 1+\exp [-(V+59-V_{sh}) / 6.2]} \\
      &\tau_{p} = 0.612 + {1 \over \exp [-(V+132.-V_{sh}) / 16.7]+\exp [(V+16.8-V_{sh}) / 18.2]} \\
      {dq \over dt} &= {\phi_q \cdot (q_{\infty}-q) \over \tau_q} \\
      &q_{\infty} = {1 \over 1+\exp [(V+83-V_{sh}) / 4]} \\
      & \begin{array}{l} \tau_{q} = \exp \left(\frac{V+467-V_{sh}}{66.6}\right)  \quad V< (-80 +V_{sh})\, mV  \\
          \tau_{q} = \exp \left(\frac{V+22-V_{sh}}{-10.5}\right)+28 \quad V \geq (-80 + V_{sh})\, mV \end{array}

  where :math:`\phi_p = 3.55^{\frac{T-24}{10}}` and :math:`\phi_q = 3^{\frac{T-24}{10}}`
  are temperature-dependent factors (:math:`T` is the temperature in Celsius),
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  T : float, Array
    The temperature.
  T_base_p : float, Array
    The base temperature factor of :math:`p` channel.
  T_base_q : float, Array
    The base temperature factor of :math:`q` channel.
  g_max : float, Array, Callable, Initializer
    The maximum conductance.
  V_sh : float, Array, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------

  .. [1] Huguenard JR, McCormick DA (1992) Simulation of the currents involved in
         rhythmic oscillations in thalamic relay neurons. J Neurophysiol 68:1373–1383.

  See Also
  --------
  ICa_p2q_form
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      T: Union[float, Array] = 36.,
      T_base_p: Union[float, Array] = 3.55,
      T_base_q: Union[float, Array] = 3.,
      g_max: Union[float, Array, Initializer, Callable] = 2.,
      V_sh: Union[float, Array, Initializer, Callable] = -3.,
      phi_p: Union[float, Array, Initializer, Callable] = None,
      phi_q: Union[float, Array, Initializer, Callable] = None,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    phi_p = T_base_p ** ((T - 24) / 10) if phi_p is None else phi_p
    phi_q = T_base_q ** ((T - 24) / 10) if phi_q is None else phi_q
    super(ICaT_HM1992, self).__init__(size,
                                      keep_size=keep_size,
                                      name=name,
                                      method=method,
                                      g_max=g_max,
                                      phi_p=phi_p,
                                      phi_q=phi_q,
                                      mode=mode)

    # parameters
    self.T = parameter(T, self.varshape, allow_none=False)
    self.T_base_p = parameter(T_base_p, self.varshape, allow_none=False)
    self.T_base_q = parameter(T_base_q, self.varshape, allow_none=False)
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    return 1. / (1 + bm.exp(-(V + 59. - self.V_sh) / 6.2))

  def f_p_tau(self, V):
    return 1. / (bm.exp(-(V + 132. - self.V_sh) / 16.7) +
                 bm.exp((V + 16.8 - self.V_sh) / 18.2)) + 0.612

  def f_q_inf(self, V):
    return 1. / (1. + bm.exp((V + 83. - self.V_sh) / 4.0))

  def f_q_tau(self, V):
    return bm.where(V >= (-80. + self.V_sh),
                    bm.exp(-(V + 22. - self.V_sh) / 10.5) + 28.,
                    bm.exp((V + 467. - self.V_sh) / 66.6))


class ICaT_HP1992(ICa_p2q_ss):
  r"""The low-threshold T-type calcium current model for thalamic
  reticular nucleus proposed by (Huguenard & Prince, 1992) [1]_.

  The dynamics of the low-threshold T-type calcium current model in thalamic
  reticular nucleus neurons [1]_ is given by:

  .. math::

      I_{CaT} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= {\phi_p \cdot (p_{\infty}-p)\over \tau_p} \\
      &p_{\infty} = {1 \over 1+\exp [-(V+52-V_{sh}) / 7.4]}  \\
      &\tau_{p} = 3+{1 \over \exp [(V+27-V_{sh}) / 10]+\exp [-(V+102-V_{sh}) / 15]} \\
      {dq \over dt} &= {\phi_q \cdot (q_{\infty}-q) \over \tau_q} \\
      &q_{\infty} = {1 \over 1+\exp [(V+80-V_{sh}) / 5]} \\
      & \tau_q = 85+ {1 \over \exp [(V+48-V_{sh}) / 4]+\exp [-(V+407-V_{sh}) / 50]}

  where :math:`\phi_p = 5^{\frac{T-24}{10}}` and :math:`\phi_q = 3^{\frac{T-24}{10}}`
  are temperature-dependent factors (:math:`T` is the temperature in Celsius),
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  T : float, Array
    The temperature.
  T_base_p : float, Array
    The base temperature factor of :math:`p` channel.
  T_base_q : float, Array
    The base temperature factor of :math:`q` channel.
  g_max : float, Array, Callable, Initializer
    The maximum conductance.
  V_sh : float, Array, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------

  .. [1] Huguenard JR, Prince DA (1992) A novel T-type current underlies
         prolonged Ca2+- dependent burst firing in GABAergic neurons of rat
         thalamic reticular nucleus. J Neurosci 12: 3804–3817.

  See Also
  --------
  ICa_p2q_form
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      T: Union[float, Array] = 36.,
      T_base_p: Union[float, Array] = 5.,
      T_base_q: Union[float, Array] = 3.,
      g_max: Union[float, Array, Initializer, Callable] = 1.75,
      V_sh: Union[float, Array, Initializer, Callable] = -3.,
      phi_p: Union[float, Array, Initializer, Callable] = None,
      phi_q: Union[float, Array, Initializer, Callable] = None,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    phi_p = T_base_p ** ((T - 24) / 10) if phi_p is None else phi_p
    phi_q = T_base_q ** ((T - 24) / 10) if phi_q is None else phi_q
    super(ICaT_HP1992, self).__init__(size,
                                      keep_size=keep_size,
                                      name=name,
                                      method=method,
                                      g_max=g_max,
                                      phi_p=phi_p,
                                      phi_q=phi_q,
                                      mode=mode)

    # parameters
    self.T = parameter(T, self.varshape, allow_none=False)
    self.T_base_p = parameter(T_base_p, self.varshape, allow_none=False)
    self.T_base_q = parameter(T_base_q, self.varshape, allow_none=False)
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    return 1. / (1. + bm.exp(-(V + 52. - self.V_sh) / 7.4))

  def f_p_tau(self, V):
    return 3. + 1. / (bm.exp((V + 27. - self.V_sh) / 10.) +
                      bm.exp(-(V + 102. - self.V_sh) / 15.))

  def f_q_inf(self, V):
    return 1. / (1. + bm.exp((V + 80. - self.V_sh) / 5.))

  def f_q_tau(self, V):
    return 85. + 1. / (bm.exp((V + 48. - self.V_sh) / 4.) +
                       bm.exp(-(V + 407. - self.V_sh) / 50.))


class ICaHT_HM1992(ICa_p2q_ss):
  r"""The high-threshold T-type calcium current model proposed by (Huguenard & McCormick, 1992) [1]_.

  The high-threshold T-type calcium current model is adopted from [1]_.
  Its dynamics is given by

  .. math::

      \begin{aligned}
      I_{\mathrm{Ca/HT}} &= g_{\mathrm{max}} p^2 q (V-E_{Ca})
      \\
      {dp \over dt} &= {\phi_{p} \cdot (p_{\infty} - p) \over \tau_{p}} \\
      &\tau_{p} =\frac{1}{\exp \left(\frac{V+132-V_{sh}}{-16.7}\right)+\exp \left(\frac{V+16.8-V_{sh}}{18.2}\right)}+0.612 \\
      & p_{\infty} = {1 \over 1+exp[-(V+59-V_{sh}) / 6.2]}
      \\
      {dq \over dt} &= {\phi_{q} \cdot (q_{\infty} - h) \over \tau_{q}} \\
      & \begin{array}{l} \tau_q = \exp \left(\frac{V+467-V_{sh}}{66.6}\right)  \quad V< (-80 +V_{sh})\, mV  \\
      \tau_q = \exp \left(\frac{V+22-V_{sh}}{-10.5}\right)+28 \quad V \geq (-80 + V_{sh})\, mV \end{array} \\
      &q_{\infty}  = {1 \over 1+exp[(V+83 -V_{shift})/4]}
      \end{aligned}

  where :math:`phi_p = 3.55^{\frac{T-24}{10}}` and :math:`phi_q = 3^{\frac{T-24}{10}}`
  are temperature-dependent factors (:math:`T` is the temperature in Celsius),
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  T : float, Array
    The temperature.
  T_base_p : float, Array
    The base temperature factor of :math:`p` channel.
  T_base_q : float, Array
    The base temperature factor of :math:`q` channel.
  g_max : float, Array, Initializer, Callable
    The maximum conductance.
  V_sh : float, Array, Initializer, Callable
    The membrane potential shift.

  References
  ----------
  .. [1] Huguenard JR, McCormick DA (1992) Simulation of the currents involved in
         rhythmic oscillations in thalamic relay neurons. J Neurophysiol 68:1373–1383.

  See Also
  --------
  ICa_p2q_form
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      T: Union[float, Array] = 36.,
      T_base_p: Union[float, Array] = 3.55,
      T_base_q: Union[float, Array] = 3.,
      g_max: Union[float, Array, Initializer, Callable] = 2.,
      V_sh: Union[float, Array, Initializer, Callable] = 25.,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    super(ICaHT_HM1992, self).__init__(size,
                                       keep_size=keep_size,
                                       name=name,
                                       method=method,
                                       g_max=g_max,
                                       phi_p=T_base_p ** ((T - 24) / 10),
                                       phi_q=T_base_q ** ((T - 24) / 10),
                                       mode=mode)

    # parameters
    self.T = parameter(T, self.varshape, allow_none=False)
    self.T_base_p = parameter(T_base_p, self.varshape, allow_none=False)
    self.T_base_q = parameter(T_base_q, self.varshape, allow_none=False)
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)

    # variables
    self.p = variable(bm.zeros, mode, self.varshape)
    self.q = variable(bm.zeros, mode, self.varshape)

    # function
    self.integral = odeint(JointEq([self.dp, self.dq]), method=method)

  def f_p_inf(self, V):
    return 1. / (1. + bm.exp(-(V + 59. - self.V_sh) / 6.2))

  def f_p_tau(self, V):
    return 1. / (bm.exp(-(V + 132. - self.V_sh) / 16.7) +
                 bm.exp((V + 16.8 - self.V_sh) / 18.2)) + 0.612

  def f_q_inf(self, V):
    return 1. / (1. + bm.exp((V + 83. - self.V_sh) / 4.))

  def f_q_tau(self, V):
    return bm.where(V >= (-80. + self.V_sh),
                    bm.exp(-(V + 22. - self.V_sh) / 10.5) + 28.,
                    bm.exp((V + 467. - self.V_sh) / 66.6))


class ICaHT_Re1993(ICa_p2q_markov):
  r"""The high-threshold T-type calcium current model proposed by (Reuveni, et al., 1993) [1]_.

  HVA Calcium current was described for neocortical neurons by Sayer et al. (1990).
  Its dynamics is given by (the rate functions are measured under 36 Celsius):

  .. math::

     \begin{aligned}
      I_{L} &=\bar{g}_{L} q^{2} r\left(V-E_{\mathrm{Ca}}\right) \\
      \frac{\mathrm{d} q}{\mathrm{~d} t} &= \phi_p (\alpha_{q}(V)(1-q)-\beta_{q}(V) q) \\
      \frac{\mathrm{d} r}{\mathrm{~d} t} &= \phi_q (\alpha_{r}(V)(1-r)-\beta_{r}(V) r) \\
      \alpha_{q} &=\frac{0.055(-27-V+V_{sh})}{\exp [(-27-V+V_{sh}) / 3.8]-1} \\
      \beta_{q} &=0.94 \exp [(-75-V+V_{sh}) / 17] \\
      \alpha_{r} &=0.000457 \exp [(-13-V+V_{sh}) / 50] \\
      \beta_{r} &=\frac{0.0065}{\exp [(-15-V+V_{sh}) / 28]+1},
      \end{aligned}

  Parameters
  ----------
  size: int, tuple of int
    The size of the simulation target.
  keep_size: bool
    Keep size or flatten the size?
  method: str
    The numerical method
  name: str
    The name of the object.
  g_max : float, Array, Callable, Initializer
    The maximum conductance.
  V_sh : float, Array, Callable, Initializer
    The membrane potential shift.
  T : float, Array
    The temperature.
  T_base_p : float, Array
    The base temperature factor of :math:`p` channel.
  T_base_q : float, Array
    The base temperature factor of :math:`q` channel.
  phi_p : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`p`.
    If `None`, :math:`\phi_p = \mathrm{T_base_p}^{\frac{T-23}{10}}`.
  phi_q : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`q`.
    If `None`, :math:`\phi_q = \mathrm{T_base_q}^{\frac{T-23}{10}}`.

  References
  ----------
  .. [1] Reuveni, I., et al. "Stepwise repolarization from Ca2+ plateaus
         in neocortical pyramidal cells: evidence for nonhomogeneous
         distribution of HVA Ca2+ channels in dendrites." Journal of
         Neuroscience 13.11 (1993): 4609-4621.

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      T: Union[float, Array] = 36.,
      T_base_p: Union[float, Array] = 2.3,
      T_base_q: Union[float, Array] = 2.3,
      phi_p: Union[float, Array, Initializer, Callable] = None,
      phi_q: Union[float, Array, Initializer, Callable] = None,
      g_max: Union[float, Array, Initializer, Callable] = 1.,
      V_sh: Union[float, Array, Initializer, Callable] = 0.,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    phi_p = T_base_p ** ((T - 23.) / 10.) if phi_p is None else phi_p
    phi_q = T_base_q ** ((T - 23.) / 10.) if phi_q is None else phi_q
    super(ICaHT_Re1993, self).__init__(size,
                                       keep_size=keep_size,
                                       name=name,
                                       method=method,
                                       g_max=g_max,
                                       phi_p=phi_p,
                                       phi_q=phi_q,
                                       mode=mode)
    self.T = parameter(T, self.varshape, allow_none=False)
    self.T_base_p = parameter(T_base_p, self.varshape, allow_none=False)
    self.T_base_q = parameter(T_base_q, self.varshape, allow_none=False)
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)

  def f_p_alpha(self, V):
    temp = -27 - V + self.V_sh
    return 0.055 * temp / (bm.exp(temp / 3.8) - 1)

  def f_p_beta(self, V):
    return 0.94 * bm.exp((-75. - V + self.V_sh) / 17.)

  def f_q_alpha(self, V):
    return 0.000457 * bm.exp((-13. - V + self.V_sh) / 50.)

  def f_q_beta(self, V):
    return 0.0065 / (bm.exp((-15. - V + self.V_sh) / 28.) + 1.)


class ICaL_IS2008(ICa_p2q_ss):
  r"""The L-type calcium channel model proposed by (Inoue & Strowbridge, 2008) [1]_.

  The L-type calcium channel model is adopted from (Inoue, et, al., 2008) [1]_.
  Its dynamics is given by:

  .. math::

      I_{CaL} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= {\phi_p \cdot (p_{\infty}-p)\over \tau_p} \\
      & p_{\infty} = {1 \over 1+\exp [-(V+10-V_{sh}) / 4.]} \\
      & \tau_{p} = 0.4+{0.7 \over \exp [(V+5-V_{sh}) / 15]+\exp [-(V+5-V_{sh}) / 15]} \\
      {dq \over dt} &= {\phi_q \cdot (q_{\infty}-q) \over \tau_q} \\
      & q_{\infty} = {1 \over 1+\exp [(V+25-V_{sh}) / 2]} \\
      & \tau_q = 300 + {100 \over \exp [(V+40-V_{sh}) / 9.5]+\exp [-(V+40-V_{sh}) / 9.5]}

  where :math:`phi_p = 3.55^{\frac{T-24}{10}}` and :math:`phi_q = 3^{\frac{T-24}{10}}`
  are temperature-dependent factors (:math:`T` is the temperature in Celsius),
  :math:`E_{Ca}` is the reversal potential of Calcium channel.

  Parameters
  ----------
  T : float
    The temperature.
  T_base_p : float
    The base temperature factor of :math:`p` channel.
  T_base_q : float
    The base temperature factor of :math:`q` channel.
  g_max : float
    The maximum conductance.
  V_sh : float
    The membrane potential shift.

  References
  ----------

  .. [1] Inoue, Tsuyoshi, and Ben W. Strowbridge. "Transient activity induces a long-lasting
         increase in the excitability of olfactory bulb interneurons." Journal of
         neurophysiology 99, no. 1 (2008): 187-199.

  See Also
  --------
  ICa_p2q_form
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      T: Union[float, Array, Initializer, Callable] = 36.,
      T_base_p: Union[float, Array, Initializer, Callable] = 3.55,
      T_base_q: Union[float, Array, Initializer, Callable] = 3.,
      g_max: Union[float, Array, Initializer, Callable] = 1.,
      V_sh: Union[float, Array, Initializer, Callable] = 0.,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    super(ICaL_IS2008, self).__init__(size,
                                      keep_size=keep_size,
                                      name=name,
                                      method=method,
                                      g_max=g_max,
                                      phi_p=T_base_p ** ((T - 24) / 10),
                                      phi_q=T_base_q ** ((T - 24) / 10),
                                      mode=mode)

    # parameters
    self.T = parameter(T, self.varshape, allow_none=False)
    self.T_base_p = parameter(T_base_p, self.varshape, allow_none=False)
    self.T_base_q = parameter(T_base_q, self.varshape, allow_none=False)
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    return 1. / (1 + bm.exp(-(V + 10. - self.V_sh) / 4.))

  def f_p_tau(self, V):
    return 0.4 + .7 / (bm.exp(-(V + 5. - self.V_sh) / 15.) +
                       bm.exp((V + 5. - self.V_sh) / 15.))

  def f_q_inf(self, V):
    return 1. / (1. + bm.exp((V + 25. - self.V_sh) / 2.))

  def f_q_tau(self, V):
    return 300. + 100. / (bm.exp((V + 40 - self.V_sh) / 9.5) +
                          bm.exp(-(V + 40 - self.V_sh) / 9.5))
