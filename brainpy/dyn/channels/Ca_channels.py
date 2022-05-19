# -*- coding: utf-8 -*-

from typing import Union, Callable

import brainpy.math as bm
from brainpy.dyn.base import Container, ConNeuGroup
from brainpy.initialize import OneInit, Initializer, init_param
from brainpy.integrators.joint_eq import JointEq
from brainpy.integrators.ode import odeint
from brainpy.types import Shape, Tensor
from .base import Ion, IonChannel

__all__ = [
  'Calcium',
  'CalciumFixed',
  'CalciumDetailed',
  'CalciumAbstract',

  'CalciumChannel',
  'IAHP',
  'ICaN',
  'ICaT',
  'ICaT_RE',
  'ICaHT',
  'ICaL',
]


class Calcium(Ion, Container):
  """The base calcium dynamics.

  Parameters
  ----------
  size: int, sequence of int
    The size of the simulation target.
  method: str
    The numerical integration method.
  name: str
    The name of the object.
  **channels
    The calcium dependent channels.
  """

  '''The type of the master object.'''
  master_type = ConNeuGroup

  """Reversal potential."""
  E: Union[float, bm.Variable, bm.JaxArray]

  """Calcium concentration."""
  C: Union[float, bm.Variable, bm.JaxArray]

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      method: str = 'exp_auto',
      name: str = None,
      **channels
  ):
    Ion.__init__(self, size, keep_size=keep_size)
    Container.__init__(self, name=name, **channels)
    self.method = method

  def current(self, V, C_Ca=None, E_Ca=None):
    C_Ca = self.C if C_Ca is None else C_Ca
    E_Ca = self.E if E_Ca is None else E_Ca
    nodes = list(self.implicit_nodes.unique().values())
    current = nodes[0].current(V, C_Ca, E_Ca)
    for node in nodes[1:]:
      current += node.current(V, C_Ca, E_Ca)
    return current


class CalciumFixed(Calcium):
  """Fixed Calcium dynamics.

  This calcium model has no dynamics. It only holds a fixed reversal potential :math:`E`.
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Tensor, Initializer, Callable] = 120.,
      C: Union[float, Tensor, Initializer, Callable] = 0.05,
      method: str = 'exp_auto',
      name: str = None,
      **channels
  ):
    super(CalciumFixed, self).__init__(size, keep_size=keep_size, method=method, name=name, **channels)
    self.E = init_param(E, self.var_shape, allow_none=False)
    self.C = init_param(C, self.var_shape, allow_none=False)

  def update(self, t, dt, V):
    for node in self.implicit_nodes.values():
      node.update(t, dt, V, self.C, self.E)

  def reset(self, V, C_Ca=None, E_Ca=None):
    C_Ca = self.C if C_Ca is None else C_Ca
    E_Ca = self.E if E_Ca is None else E_Ca
    for node in self.implicit_nodes.values():
      node.reset(V, C_Ca, E_Ca)


class CalciumDetailed(Calcium):
  r"""Dynamical Calcium model.

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
  C_0 : float
    The :math:`Ca ^{2+}` concentration outside of the membrane.
  R : float
    The gas constant. (:math:` J*mol^{-1}*K^{-1}`)

  References
  ----------

  .. [1] Destexhe, Alain, Agnessa Babloyantz, and Terrence J. Sejnowski. "Ionic mechanisms for intrinsic slow oscillations in thalamic relay neurons." Biophysical journal 65, no. 4 (1993): 1538-1552.
  .. [2] Bazhenov, Maxim, Igor Timofeev, Mircea Steriade, and Terrence J. Sejnowski. "Cellular and network models for intrathalamic augmenting responses during 10-Hz stimulation." Journal of neurophysiology 79, no. 5 (1998): 2730-2748.

  """

  R = 8.31441  # gas constant, J*mol-1*K-1
  F = 96.489  # the Faraday constant

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      d: Union[float, Tensor, Initializer, Callable] = 1.,
      C_rest: Union[float, Tensor, Initializer, Callable] = 0.05,
      tau: Union[float, Tensor, Initializer, Callable] = 5.,
      C_0: Union[float, Tensor, Initializer, Callable] = 2.,
      T: Union[float, Tensor, Initializer, Callable] = 36.,
      C_initializer: Union[Initializer, Callable, Tensor] = OneInit(0.05),
      E_initializer: Union[Initializer, Callable, Tensor] = OneInit(120.),
      method: str = 'exp_auto',
      name: str = None,
      **channels
  ):
    super(CalciumDetailed, self).__init__(size,
                                          keep_size=keep_size,
                                          method=method,
                                          name=name, **channels)

    # parameters
    self.T = init_param(T, self.var_shape, allow_none=False)  # temperature
    self.d = init_param(d, self.var_shape, allow_none=False)
    self.tau = init_param(tau, self.var_shape, allow_none=False)
    self.C_rest = init_param(C_rest, self.var_shape, allow_none=False)
    self.C_0 = init_param(C_0, self.var_shape, allow_none=False)
    self._E_initializer = E_initializer
    self._C_initializer = C_initializer

    # variables
    self.C = bm.Variable(init_param(C_initializer, self.var_shape))  # Calcium concentration
    self.E = bm.Variable(init_param(E_initializer, self.var_shape))  # Reversal potential

    # function
    self.integral = odeint(self.derivative, method=method)

  def reset(self, V, C_Ca=None, E_Ca=None):
    self.C[:] = init_param(self._C_initializer, self.var_shape) if (C_Ca is None) else C_Ca
    self.E[:] = init_param(self._E_initializer, self.var_shape) if (E_Ca is None) else E_Ca
    for node in self.implicit_nodes.values():
      node.reset(V, self.C, self.E)

  def derivative(self, C, t, V):
    ICa = self.current(V, C, self.E)
    return - ICa / (2 * self.F * self.d) + (self.C_rest - C) / self.tau

  def update(self, t, dt, V):
    C = self.integral(self.C.value, t, V, dt)
    for node in self.implicit_nodes.values():
      node.update(t, dt, V, self.C, self.E)
    self.E.value = self.R * (273.15 + self.T) / (2 * self.F) * bm.log(self.C_0 / C)
    self.C.value = C


class CalciumAbstract(Calcium):
  r"""The first-order calcium concentration model.

  .. math::

     Ca' = -\alpha I_{Ca} + -\beta Ca



  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      alpha: Union[float, Tensor, Initializer, Callable] = 0.13,
      beta: Union[float, Tensor, Initializer, Callable] = 0.075,
      C_initializer: Union[Initializer, Callable, Tensor] = OneInit(0.05),
      E_initializer: Union[Initializer, Callable, Tensor] = OneInit(120.),
      method: str = 'exp_auto',
      name: str = None
  ):
    super(CalciumAbstract, self).__init__(size, keep_size=keep_size, name=name)

    # parameters
    self.alpha = init_param(alpha, self.var_shape, allow_none=False)
    self.beta = init_param(beta, self.var_shape, allow_none=False)

    # variables
    self.C = bm.Variable(init_param(C_initializer, self.var_shape))  # Calcium concentration
    self.E = bm.Variable(init_param(E_initializer, self.var_shape))  # Reversal potential

    # functions
    self.integral = odeint(self.derivative, method=method)

  def reset(self, V, C_Ca=None, E_Ca=None):
    self.C[:] = init_param(self._C_initializer, self.var_shape) if (C_Ca is None) else C_Ca
    self.E[:] = init_param(self._E_initializer, self.var_shape) if (E_Ca is None) else E_Ca
    for node in self.implicit_nodes.values():
      node.reset(V, self.C, self.E)

  def derivative(self, C, t, V):
    ICa = self.current(V, C, self.E)
    return - self.alpha * ICa - self.beta * C

  def update(self, t, dt, V):
    C = self.integral(self.C.value, t, V, dt)
    for node in self.implicit_nodes.values():
      node.update(t, dt, V, self.C, self.E)
    self.E.value = self.R * (273.15 + self.T) / (2 * self.F) * bm.log(self.C_0 / C)
    self.C.value = C


# -------------------------


class CalciumChannel(IonChannel):
  """Base class for Calcium ion channels."""

  '''The type of the master object.'''
  master_master_type = Calcium

  def update(self, t, dt, V, C_Ca, E_Ca):
    raise NotImplementedError

  def current(self, V, C_Ca, E_Ca):
    raise NotImplementedError

  def reset(self, V, C_Ca, E_Ca):
    raise NotImplementedError


class IAHP(CalciumChannel):
  r"""The calcium-dependent potassium current model.

  The dynamics of the calcium-dependent potassium current model is given by:

  .. math::

      \begin{aligned}
      I_{AHP} &= g_{\mathrm{max}} p (V - E) \\
      {dp \over dt} &= {p_{\infty}(V) - p \over \tau_p(V)} \\
      p_{\infty} &=\frac{48[Ca^{2+}]_i}{\left(48[Ca^{2+}]_i +0.09\right)} \\
      \tau_p &=\frac{1}{\left(48[Ca^{2+}]_i +0.09\right)}
      \end{aligned}

  where :math:`E` is the reversal potential, :math:`g_{max}` is the maximum conductance.


  Parameters
  ----------
  g_max : float
    The maximal conductance density (:math:`mS/cm^2`).
  E : float
    The reversal potential (mV).

  References
  ----------

  .. [1] Contreras, D., R. Curró Dossi, and M. Steriade. "Electrophysiological
         properties of cat reticular thalamic neurones in vivo." The Journal of
         Physiology 470.1 (1993): 273-294.
  .. [2] Mulle, Ch, Anamaria Madariaga, and M. Deschênes. "Morphology and
         electrophysiological properties of reticularis thalami neurons in
         cat: in vivo study of a thalamic pacemaker." Journal of
         Neuroscience 6.8 (1986): 2134-2145.
  .. [3] Avanzini, G., et al. "Intrinsic properties of nucleus reticularis
         thalami neurones of the rat studied in vitro." The Journal of
         Physiology 416.1 (1989): 111-122.
  .. [4] Destexhe, Alain, et al. "A model of spindle rhythmicity in the isolated
         thalamic reticular nucleus." Journal of neurophysiology 72.2 (1994): 803-818.
  .. [5] Vijayan S, Kopell NJ (2012) Thalamic model of awake alpha oscillations and
         implications for stimulus processing. Proc Natl Acad Sci USA 109: 18553–18558.

  """

  '''The type of the master object.'''
  master_master_type = CalciumDetailed

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Tensor, Initializer, Callable] = -80.,
      g_max: Union[float, Tensor, Initializer, Callable] = 1.,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(IAHP, self).__init__(size, keep_size=keep_size,  name=name)

    # parameters
    self.E = init_param(E, self.var_shape, allow_none=False)
    self.g_max = init_param(g_max, self.var_shape, allow_none=False)

    # variables
    self.p = bm.Variable(bm.zeros(self.var_shape))

    # function
    self.integral = odeint(self.derivative, method=method)

  def derivative(self, p, t, V, C_Ca, E_Ca):
    C2 = 48 * C_Ca ** 2
    C3 = C2 + 0.09
    return (C2 / C3 - p) * C3

  def update(self, t, dt, V, C_Ca, E_Ca):
    self.p.value = self.integral(self.p, t, C=C_Ca, dt=dt)

  def current(self, V, C_Ca, E_Ca):
    return self.g_max * self.p * (self.E - V)

  def reset(self, V, C_Ca, E_Ca):
    C2 = 48 * C_Ca ** 2
    C3 = C2 + 0.09
    self.p.value = C2 / C3


class ICaN(CalciumChannel):
  r"""The calcium-activated non-selective cation channel model.

  The dynamics of the calcium-activated non-selective cation channel model is given by:

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
  master_master_type = CalciumDetailed

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Tensor, Initializer, Callable] = 10.,
      g_max: Union[float, Tensor, Initializer, Callable] = 1.,
      phi: Union[float, Tensor, Initializer, Callable] = 1.,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(ICaN, self).__init__(size, keep_size=keep_size, name=name)

    # parameters
    self.E = init_param(E, self.var_shape, allow_none=False)
    self.g_max = init_param(g_max, self.var_shape, allow_none=False)
    self.phi = init_param(phi, self.var_shape, allow_none=False)

    # variables
    self.p = bm.Variable(bm.zeros(self.var_shape))

    # function
    self.integral = odeint(self.derivative, method=method)

  def derivative(self, p, t, V):
    phi_p = 1.0 / (1 + bm.exp(-(V + 43.) / 5.2))
    p_inf = 2.7 / (bm.exp(-(V + 55.) / 15.) + bm.exp((V + 55.) / 15.)) + 1.6
    return self.phi * (phi_p - p) / p_inf

  def update(self, t, dt, V, C_Ca, E_Ca):
    self.p.value = self.integral(self.p, t, V, dt)

  def current(self, V, C_Ca, E_Ca):
    M = C_Ca / (C_Ca + 0.2)
    g = self.g_max * M * self.p
    return g * (self.E - V)

  def reset(self, V, C_Ca, E_Ca):
    self.p.value = 1.0 / (1 + bm.exp(-(V + 43.) / 5.2))


class ICaT(CalciumChannel):
  r"""The low-threshold T-type calcium current model.

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

  .. [1] Huguenard JR, McCormick DA (1992) Simulation of the currents involved in
         rhythmic oscillations in thalamic relay neurons. J Neurophysiol 68:1373–1383.
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      T: Union[float, Tensor, Initializer, Callable] = 36.,
      T_base_p: Union[float, Tensor, Initializer, Callable] = 3.55,
      T_base_q: Union[float, Tensor, Initializer, Callable] = 3.,
      g_max: Union[float, Tensor, Initializer, Callable] = 2.,
      V_sh: Union[float, Tensor, Initializer, Callable] = -3.,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(ICaT, self).__init__(size, keep_size=keep_size, name=name)

    # parameters
    self.T = init_param(T, self.var_shape, allow_none=False)
    self.T_base_p = init_param(T_base_p, self.var_shape, allow_none=False)
    self.T_base_q = init_param(T_base_q, self.var_shape, allow_none=False)
    self.g_max = init_param(g_max, self.var_shape, allow_none=False)
    self.V_sh = init_param(V_sh, self.var_shape, allow_none=False)
    self.phi_p = self.T_base_p ** ((self.T - 24) / 10)
    self.phi_q = self.T_base_q ** ((self.T - 24) / 10)

    # variables
    self.p = bm.Variable(bm.zeros(self.var_shape))
    self.q = bm.Variable(bm.zeros(self.var_shape))

    # functions
    self.integral = odeint(JointEq([self.dp, self.dq]), method=method)

  def dp(self, p, t, V):
    p_inf = 1. / (1 + bm.exp(-(V + 59. - self.V_sh) / 6.2))
    p_tau = 1. / (bm.exp(-(V + 132. - self.V_sh) / 16.7) + bm.exp((V + 16.8 - self.V_sh) / 18.2)) + 0.612
    return self.phi_p * (p_inf - p) / p_tau

  def dq(self, q, t, V):
    q_inf = 1. / (1. + bm.exp((V + 83. - self.V_sh) / 4.0))
    q_tau = bm.where(V >= (-80. + self.V_sh),
                     bm.exp(-(V + 22. - self.V_sh) / 10.5) + 28.,
                     bm.exp((V + 467. - self.V_sh) / 66.6))
    return self.phi_q * (q_inf - q) / q_tau

  def update(self, t, dt, V, C_Ca, E_Ca):
    self.p.value, self.q.value = self.integral(self.p, self.q, t, V, dt)

  def current(self, V, C_Ca, E_Ca):
    return self.g_max * self.p * self.p * self.q * (E_Ca - V)

  def reset(self, V, C_Ca, E_Ca):
    self.p.value = 1. / (1 + bm.exp(-(V + 59. - self.V_sh) / 6.2))
    self.q.value = 1. / (1. + bm.exp((V + 83. - self.V_sh) / 4.0))


class ICaT_RE(CalciumChannel):
  r"""The low-threshold T-type calcium current model in thalamic reticular nucleus.

  The dynamics of the low-threshold T-type calcium current model [1]_ [2]_ in thalamic
  reticular nucleus neurons is given by:

  .. math::

      I_{CaT} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= {\phi_p \cdot (p_{\infty}-p)\over \tau_p} \\
      &p_{\infty} = {1 \over 1+\exp [-(V+52-V_{sh}) / 7.4]}  \\
      &\tau_{p} = 3+{1 \over \exp [(V+27-V_{sh}) / 10]+\exp [-(V+102-V_{sh}) / 15]} \\
      {dq \over dt} &= {\phi_q \cdot (q_{\infty}-q) \over \tau_q} \\
      &q_{\infty} = {1 \over 1+\exp [(V+80-V_{sh}) / 5]} \\
      & \tau_q = 85+ {1 \over \exp [(V+48-V_{sh}) / 4]+\exp [-(V+407-V_{sh}) / 50]}

  where :math:`phi_p = 5^{\frac{T-24}{10}}` and :math:`phi_q = 3^{\frac{T-24}{10}}`
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

  .. [1] Avanzini, G., et al. "Intrinsic properties of nucleus reticularis thalami
         neurones of the rat studied in vitro." The Journal of
         Physiology 416.1 (1989): 111-122.
  .. [2] Bal, Thierry, and DAVID A. McCORMICK. "Mechanisms of oscillatory activity
         in guinea‐pig nucleus reticularis thalami in vitro: a mammalian
         pacemaker." The Journal of Physiology 468.1 (1993): 669-691.

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      T: Union[float, Tensor, Initializer, Callable] = 36.,
      T_base_p: Union[float, Tensor, Initializer, Callable] = 5.,
      T_base_q: Union[float, Tensor, Initializer, Callable] = 3.,
      g_max: Union[float, Tensor, Initializer, Callable] = 1.75,
      V_sh: Union[float, Tensor, Initializer, Callable] = -3.,
      method='exp_auto',
      name=None
  ):
    super(ICaT_RE, self).__init__(size, keep_size=keep_size, name=name)

    # parameters
    self.T = init_param(T, self.var_shape, allow_none=False)
    self.T_base_p = init_param(T_base_p, self.var_shape, allow_none=False)
    self.T_base_q = init_param(T_base_q, self.var_shape, allow_none=False)
    self.g_max = init_param(g_max, self.var_shape, allow_none=False)
    self.V_sh = init_param(V_sh, self.var_shape, allow_none=False)
    self.phi_p = self.T_base_p ** ((self.T - 24) / 10)
    self.phi_q = self.T_base_q ** ((self.T - 24) / 10)

    # variables
    self.p = bm.Variable(bm.zeros(self.var_shape))
    self.q = bm.Variable(bm.zeros(self.var_shape))

    # function
    self.integral = odeint(JointEq([self.dp, self.dq]), method=method)

  def dp(self, p, t, V):
    p_inf = 1. / (1. + bm.exp(-(V + 52. - self.V_sh) / 7.4))
    p_tau = 3. + 1. / (bm.exp((V + 27. - self.V_sh) / 10.) + bm.exp(-(V + 102. - self.V_sh) / 15.))
    return self.phi_p * (p_inf - p) / p_tau

  def dq(self, q, t, V):
    q_inf = 1. / (1. + bm.exp((V + 80. - self.V_sh) / 5.))
    q_tau = 85. + 1. / (bm.exp((V + 48. - self.V_sh) / 4.) + bm.exp(-(V + 407. - self.V_sh) / 50.))
    return self.phi_q * (q_inf - q) / q_tau

  def update(self, t, dt, V, C_Ca, E_Ca):
    self.p.value, self.q.value = self.integral(self.p, self.q, t, V, dt)

  def current(self, V, C_Ca, E_Ca):
    return self.g_max * self.p * self.p * self.q * (E_Ca - V)

  def reset(self, V, C_Ca, E_Ca):
    self.p.value = 1. / (1. + bm.exp(-(V + 52. - self.V_sh) / 7.4))
    self.q.value = 1. / (1. + bm.exp((V + 80. - self.V_sh) / 5.))


class ICaHT(CalciumChannel):
  r"""The high-threshold T-type calcium current model.

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
  .. [1] Huguenard JR, McCormick DA (1992) Simulation of the currents involved in
         rhythmic oscillations in thalamic relay neurons. J Neurophysiol 68:1373–1383.
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      T: Union[float, Tensor, Initializer, Callable] = 36.,
      T_base_p: Union[float, Tensor, Initializer, Callable] = 3.55,
      T_base_q: Union[float, Tensor, Initializer, Callable] = 3.,
      g_max: Union[float, Tensor, Initializer, Callable] = 2.,
      V_sh: Union[float, Tensor, Initializer, Callable] = 25.,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(ICaHT, self).__init__(size, keep_size=keep_size, name=name)

    # parameters
    self.T = init_param(T, self.var_shape, allow_none=False)
    self.T_base_p = init_param(T_base_p, self.var_shape, allow_none=False)
    self.T_base_q = init_param(T_base_q, self.var_shape, allow_none=False)
    self.g_max = init_param(g_max, self.var_shape, allow_none=False)
    self.V_sh = init_param(V_sh, self.var_shape, allow_none=False)
    self.phi_p = self.T_base_p ** ((self.T - 24) / 10)
    self.phi_q = self.T_base_q ** ((self.T - 24) / 10)

    # variables
    self.p = bm.Variable(bm.zeros(self.var_shape))
    self.q = bm.Variable(bm.zeros(self.var_shape))

    # function
    self.integral = odeint(JointEq([self.dp, self.dq]), method=method)

  def dp(self, p, t, V):
    p_inf = 1. / (1. + bm.exp(-(V + 59. - self.V_sh) / 6.2))
    p_tau = 1. / (bm.exp(-(V + 132. - self.V_sh) / 16.7) + bm.exp((V + 16.8 - self.V_sh) / 18.2)) + 0.612
    return self.phi_p * (p_inf - p) / p_tau

  def dq(self, q, t, V):
    q_inf = 1. / (1. + bm.exp((V + 83. - self.V_sh) / 4.))
    q_tau = bm.where(V >= (-80. + self.V_sh),
                     bm.exp(-(V + 22. - self.V_sh) / 10.5) + 28.,
                     bm.exp((V + 467. - self.V_sh) / 66.6))
    return self.phi_q * (q_inf - q) / q_tau

  def update(self, t, dt, V, C_Ca, E_Ca):
    self.p.value, self.q.value = self.integral(self.p, self.q, t, V, dt)

  def current(self, V, C_Ca, E_Ca):
    return self.g_max * self.p * self.p * self.q * (E_Ca - V)

  def reset(self, V, C_Ca, E_Ca):
    self.p.value = 1. / (1. + bm.exp(-(V + 59. - self.V_sh) / 6.2))
    self.q.value = 1. / (1. + bm.exp((V + 83. - self.V_sh) / 4.))


class ICaL(CalciumChannel):
  r"""The L-type calcium channel model.

  The L-type calcium channel model is adopted from (Inoue, et, al., 2008) [1]_.
  Its dynamics is given by:

  .. math::

      I_{CaL} &= g_{max} p^2 q(V-E_{Ca}) \\
      {dp \over dt} &= {\phi_p \cdot (p_{\infty}-p)\over \tau_p} \\
      &p_{\infty} = {1 \over 1+\exp [-(V+10-V_{sh}) / 4.]} \\
      &\tau_{p} = 0.4+{0.7 \over \exp [(V+5-V_{sh}) / 15]+\exp [-(V+5-V_{sh}) / 15]} \\
      {dq \over dt} &= {\phi_q \cdot (q_{\infty}-q) \over \tau_q} \\
      &q_{\infty} = {1 \over 1+\exp [(V+25-V_{sh}) / 2]} \\
      &\tau_q = 300 + {100 \over \exp [(V+40-V_{sh}) / 9.5]+\exp [-(V+40-V_{sh}) / 9.5]}

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
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      T: Union[float, Tensor, Initializer, Callable] = 36.,
      T_base_p: Union[float, Tensor, Initializer, Callable] = 3.55,
      T_base_q: Union[float, Tensor, Initializer, Callable] = 3.,
      g_max: Union[float, Tensor, Initializer, Callable] = 1.,
      V_sh: Union[float, Tensor, Initializer, Callable] = 0.,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(ICaL, self).__init__(size, keep_size=keep_size, name=name)

    # parameters
    self.T = init_param(T, self.var_shape, allow_none=False)
    self.T_base_p = init_param(T_base_p, self.var_shape, allow_none=False)
    self.T_base_q = init_param(T_base_q, self.var_shape, allow_none=False)
    self.g_max = init_param(g_max, self.var_shape, allow_none=False)
    self.V_sh = init_param(V_sh, self.var_shape, allow_none=False)
    self.phi_p = self.T_base_p ** ((self.T - 24) / 10)
    self.phi_q = self.T_base_q ** ((self.T - 24) / 10)

    # variables
    self.p = bm.Variable(bm.zeros(self.var_shape))
    self.q = bm.Variable(bm.zeros(self.var_shape))

    # function
    self.integral = odeint(JointEq([self.dp, self.dq]), method=method)

  def dp(self, p, t, V):
    p_inf = 1. / (1 + bm.exp(-(V + 10. - self.V_sh) / 4.))
    p_tau = 0.4 + .7 / (bm.exp(-(V + 5. - self.V_sh) / 15.) + bm.exp((V + 5. - self.V_sh) / 15.))
    dpdt = self.phi_p * (p_inf - p) / p_tau
    return dpdt

  def dq(self, q, t, V):
    q_inf = 1. / (1. + bm.exp((V + 25. - self.V_sh) / 2.))
    q_tau = 300. + 100. / (bm.exp((V + 40 - self.V_sh) / 9.5) + bm.exp(-(V + 40 - self.V_sh) / 9.5))
    dqdt = self.phi_q * (q_inf - q) / q_tau
    return dqdt

  def update(self, t, dt, V, C_Ca, E_Ca):
    self.p.value, self.q.value = self.integral(self.p, self.q, t, V, dt)

  def current(self, V, C_Ca, E_Ca):
    return self.g_max * self.p * self.p * self.q * (E_Ca - V)

  def reset(self, V, C_Ca, E_Ca):
    self.p.value = 1. / (1 + bm.exp(-(V + 10. - self.V_sh) / 4.))
    self.q.value = 1. / (1. + bm.exp((V + 25. - self.V_sh) / 2.))
