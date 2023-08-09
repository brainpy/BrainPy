# -*- coding: utf-8 -*-

"""
This module implements voltage-dependent calcium channels.

"""

from typing import Union, Callable, Optional

import brainpy.math as bm
from brainpy._src.context import share
from brainpy._src.dyn.ions.calcium import Calcium, CalciumDyna
from brainpy._src.initialize import Initializer, parameter, variable
from brainpy._src.integrators.joint_eq import JointEq
from brainpy._src.integrators.ode.generic import odeint
from brainpy.types import Shape, ArrayType
from .base import IonChannel

__all__ = [
  'CalciumChannel',

  'ICaN_IS2008',
  'ICaT_HM1992',
  'ICaT_HP1992',
  'ICaHT_HM1992',
  'ICaL_IS2008',
]


class CalciumChannel(IonChannel):
  """Base class for Calcium ion channels."""

  master_type = Calcium
  '''The type of the master object.'''

  def update(self, V, C, E):
    raise NotImplementedError

  def current(self, V, C, E):
    raise NotImplementedError

  def reset(self, V, C, E, batch_size: int = None):
    self.reset_state(V, C, E, batch_size)

  def reset_state(self, V, C, E, batch_size: int = None):
    raise NotImplementedError('Must be implemented by the subclass.')


class _ICa_p2q_ss(CalciumChannel):
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
  g_max : float, ArrayType, Callable, Initializer
    The maximum conductance.
  phi_p : float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      phi_p: Union[float, ArrayType, Initializer, Callable] = 3.,
      phi_q: Union[float, ArrayType, Initializer, Callable] = 3.,
      g_max: Union[float, ArrayType, Initializer, Callable] = 2.,
      method: str = 'exp_auto',
      mode: Optional[bm.Mode] = None,
      name: Optional[str] = None
  ):
    super().__init__(size,
                     keep_size=keep_size,
                     name=name,
                     mode=mode, )

    # parameters
    self.phi_p = parameter(phi_p, self.varshape, allow_none=False)
    self.phi_q = parameter(phi_q, self.varshape, allow_none=False)
    self.g_max = parameter(g_max, self.varshape, allow_none=False)

    # variables
    self.p = variable(bm.zeros, self.mode, self.varshape)
    self.q = variable(bm.zeros, self.mode, self.varshape)

    # functions
    self.integral = odeint(JointEq([self.dp, self.dq]), method=method)

  def dp(self, p, t, V):
    return self.phi_p * (self.f_p_inf(V) - p) / self.f_p_tau(V)

  def dq(self, q, t, V):
    return self.phi_q * (self.f_q_inf(V) - q) / self.f_q_tau(V)

  def update(self, V, C, E):
    self.p.value, self.q.value = self.integral(self.p, self.q, share['t'], V, share['dt'])

  def current(self, V, C, E):
    return self.g_max * self.p * self.p * self.q * (E - V)

  def reset_state(self, V, C, E, batch_size=None):
    self.p.value = self.f_p_inf(V)
    self.q.value = self.f_q_inf(V)
    if isinstance(batch_size, int):
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


class _ICa_p2q_markov(CalciumChannel):
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
  g_max : float, ArrayType, Callable, Initializer
    The maximum conductance.
  phi_p : float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`q`.

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      phi_p: Union[float, ArrayType, Initializer, Callable] = 3.,
      phi_q: Union[float, ArrayType, Initializer, Callable] = 3.,
      g_max: Union[float, ArrayType, Initializer, Callable] = 2.,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(size,
                     keep_size=keep_size,
                     name=name,
                     mode=mode)

    # parameters
    self.phi_p = parameter(phi_p, self.varshape, allow_none=False)
    self.phi_q = parameter(phi_q, self.varshape, allow_none=False)
    self.g_max = parameter(g_max, self.varshape, allow_none=False)

    # variables
    self.p = variable(bm.zeros, self.mode, self.varshape)
    self.q = variable(bm.zeros, self.mode, self.varshape)

    # functions
    self.integral = odeint(JointEq([self.dp, self.dq]), method=method)

  def dp(self, p, t, V):
    return self.phi_p * (self.f_p_alpha(V) * (1 - p) - self.f_p_beta(V) * p)

  def dq(self, q, t, V):
    return self.phi_q * (self.f_q_alpha(V) * (1 - q) - self.f_q_beta(V) * q)

  def update(self, V, C, E):
    self.p.value, self.q.value = self.integral(self.p, self.q, share['t'], V, share['dt'])

  def current(self, V, C, E):
    return self.g_max * self.p * self.p * self.q * (E - V)

  def reset_state(self, V, C, E, batch_size=None):
    alpha, beta = self.f_p_alpha(V), self.f_p_beta(V)
    self.p.value = alpha / (alpha + beta)
    alpha, beta = self.f_q_alpha(V), self.f_q_beta(V)
    self.q.value = alpha / (alpha + beta)
    if isinstance(batch_size, int):
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
      E: Union[float, ArrayType, Initializer, Callable] = 10.,
      g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
      phi: Union[float, ArrayType, Initializer, Callable] = 1.,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(size,
                     keep_size=keep_size,
                     name=name,
                     mode=mode)

    # parameters
    self.E = parameter(E, self.varshape, allow_none=False)
    self.g_max = parameter(g_max, self.varshape, allow_none=False)
    self.phi = parameter(phi, self.varshape, allow_none=False)

    # variables
    self.p = variable(bm.zeros, self.mode, self.varshape)

    # function
    self.integral = odeint(self.derivative, method=method)

  def derivative(self, p, t, V):
    phi_p = 1.0 / (1 + bm.exp(-(V + 43.) / 5.2))
    p_inf = 2.7 / (bm.exp(-(V + 55.) / 15.) + bm.exp((V + 55.) / 15.)) + 1.6
    return self.phi * (phi_p - p) / p_inf

  def update(self, V, C, E):
    self.p.value = self.integral(self.p.value, share['t'], V, share['dt'])

  def current(self, V, C, E):
    M = C / (C + 0.2)
    g = self.g_max * M * self.p
    return g * (self.E - V)

  def reset_state(self, V, C, E, batch_size=None):
    self.p.value = 1.0 / (1 + bm.exp(-(V + 43.) / 5.2))
    if isinstance(batch_size, int):
      assert self.p.shape[0] == batch_size


class ICaT_HM1992(_ICa_p2q_ss):
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
  T : float, ArrayType
    The temperature.
  T_base_p : float, ArrayType
    The brainpy_object temperature factor of :math:`p` channel.
  T_base_q : float, ArrayType
    The brainpy_object temperature factor of :math:`q` channel.
  g_max : float, ArrayType, Callable, Initializer
    The maximum conductance.
  V_sh : float, ArrayType, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, ArrayType, Callable, Initializer
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
      T: Union[float, ArrayType] = 36.,
      T_base_p: Union[float, ArrayType] = 3.55,
      T_base_q: Union[float, ArrayType] = 3.,
      g_max: Union[float, ArrayType, Initializer, Callable] = 2.,
      V_sh: Union[float, ArrayType, Initializer, Callable] = -3.,
      phi_p: Union[float, ArrayType, Initializer, Callable] = None,
      phi_q: Union[float, ArrayType, Initializer, Callable] = None,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    phi_p = T_base_p ** ((T - 24) / 10) if phi_p is None else phi_p
    phi_q = T_base_q ** ((T - 24) / 10) if phi_q is None else phi_q
    super().__init__(size,
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


class ICaT_HP1992(_ICa_p2q_ss):
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
  T : float, ArrayType
    The temperature.
  T_base_p : float, ArrayType
    The brainpy_object temperature factor of :math:`p` channel.
  T_base_q : float, ArrayType
    The brainpy_object temperature factor of :math:`q` channel.
  g_max : float, ArrayType, Callable, Initializer
    The maximum conductance.
  V_sh : float, ArrayType, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, ArrayType, Callable, Initializer
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
      T: Union[float, ArrayType] = 36.,
      T_base_p: Union[float, ArrayType] = 5.,
      T_base_q: Union[float, ArrayType] = 3.,
      g_max: Union[float, ArrayType, Initializer, Callable] = 1.75,
      V_sh: Union[float, ArrayType, Initializer, Callable] = -3.,
      phi_p: Union[float, ArrayType, Initializer, Callable] = None,
      phi_q: Union[float, ArrayType, Initializer, Callable] = None,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    phi_p = T_base_p ** ((T - 24) / 10) if phi_p is None else phi_p
    phi_q = T_base_q ** ((T - 24) / 10) if phi_q is None else phi_q
    super().__init__(size,
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


class ICaHT_HM1992(_ICa_p2q_ss):
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
  T : float, ArrayType
    The temperature.
  T_base_p : float, ArrayType
    The brainpy_object temperature factor of :math:`p` channel.
  T_base_q : float, ArrayType
    The brainpy_object temperature factor of :math:`q` channel.
  g_max : float, ArrayType, Initializer, Callable
    The maximum conductance.
  V_sh : float, ArrayType, Initializer, Callable
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
      T: Union[float, ArrayType] = 36.,
      T_base_p: Union[float, ArrayType] = 3.55,
      T_base_q: Union[float, ArrayType] = 3.,
      g_max: Union[float, ArrayType, Initializer, Callable] = 2.,
      V_sh: Union[float, ArrayType, Initializer, Callable] = 25.,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(size,
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
    self.p = variable(bm.zeros, self.mode, self.varshape)
    self.q = variable(bm.zeros, self.mode, self.varshape)

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


class ICaHT_Re1993(_ICa_p2q_markov):
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
  g_max : float, ArrayType, Callable, Initializer
    The maximum conductance.
  V_sh : float, ArrayType, Callable, Initializer
    The membrane potential shift.
  T : float, ArrayType
    The temperature.
  T_base_p : float, ArrayType
    The brainpy_object temperature factor of :math:`p` channel.
  T_base_q : float, ArrayType
    The brainpy_object temperature factor of :math:`q` channel.
  phi_p : optional, float, ArrayType, Callable, Initializer
    The temperature factor for channel :math:`p`.
    If `None`, :math:`\phi_p = \mathrm{T_base_p}^{\frac{T-23}{10}}`.
  phi_q : optional, float, ArrayType, Callable, Initializer
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
      T: Union[float, ArrayType] = 36.,
      T_base_p: Union[float, ArrayType] = 2.3,
      T_base_q: Union[float, ArrayType] = 2.3,
      phi_p: Union[float, ArrayType, Initializer, Callable] = None,
      phi_q: Union[float, ArrayType, Initializer, Callable] = None,
      g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
      V_sh: Union[float, ArrayType, Initializer, Callable] = 0.,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    phi_p = T_base_p ** ((T - 23.) / 10.) if phi_p is None else phi_p
    phi_q = T_base_q ** ((T - 23.) / 10.) if phi_q is None else phi_q
    super().__init__(size,
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


class ICaL_IS2008(_ICa_p2q_ss):
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
    The brainpy_object temperature factor of :math:`p` channel.
  T_base_q : float
    The brainpy_object temperature factor of :math:`q` channel.
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
      T: Union[float, ArrayType, Initializer, Callable] = 36.,
      T_base_p: Union[float, ArrayType, Initializer, Callable] = 3.55,
      T_base_q: Union[float, ArrayType, Initializer, Callable] = 3.,
      g_max: Union[float, ArrayType, Initializer, Callable] = 1.,
      V_sh: Union[float, ArrayType, Initializer, Callable] = 0.,
      method: str = 'exp_auto',
      name: Optional[str] = None,
      mode: Optional[bm.Mode] = None,
  ):
    super().__init__(size,
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
