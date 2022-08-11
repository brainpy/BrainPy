# -*- coding: utf-8 -*-

"""
This module implements voltage-dependent potassium channels.

"""

from typing import Union, Callable, Optional

import brainpy.math as bm
from brainpy.initialize import Initializer, parameter, variable
from brainpy.integrators import odeint, JointEq
from brainpy.types import Shape, Array
from brainpy.modes import Mode, BatchingMode, normal
from .base import PotassiumChannel

__all__ = [
  'IK_p4_markov',
  'IKDR_Ba2002',
  'IK_TM1991',
  'IK_HH1952',

  'IKA_p4q_ss',
  'IKA1_HM1992',
  'IKA2_HM1992',

  'IKK2_pq_ss',
  'IKK2A_HM1992',
  'IKK2B_HM1992',

  'IKNI_Ya1989',
]


class IK_p4_markov(PotassiumChannel):
  r"""The delayed rectifier potassium channel of :math:`p^4`
  current which described with first-order Markov chain.

  This general potassium current model should have the form of

  .. math::

      \begin{aligned}
      I_{\mathrm{K}} &= g_{\mathrm{max}} * p^4 \\
      \frac{dp}{dt} &= \phi * (\alpha_p (1-p) - \beta_p p)
      \end{aligned}

  where :math:`\phi` is a temperature-dependent factor.

  Parameters
  ----------
  size: int, sequence of int
    The object size.
  keep_size: bool
    Whether we use `size` to initialize the variable. Otherwise, variable shape
    will be initialized as `num`.
  g_max : float, JaxArray, ndarray, Initializer, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  E : float, JaxArray, ndarray, Initializer, Callable
    The reversal potential (mV).
  phi : float, JaxArray, ndarray, Initializer, Callable
    The temperature-dependent factor.
  method: str
    The numerical integration method.
  name: str
    The object name.

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Array, Initializer, Callable] = -90.,
      g_max: Union[float, Array, Initializer, Callable] = 10.,
      phi: Union[float, Array, Initializer, Callable] = 1.,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    super(IK_p4_markov, self).__init__(size,
                                       keep_size=keep_size,
                                       name=name,
                                       mode=mode)

    self.E = parameter(E, self.varshape, allow_none=False)
    self.g_max = parameter(g_max, self.varshape, allow_none=False)
    self.phi = parameter(phi, self.varshape, allow_none=False)

    # variables
    self.p = variable(bm.zeros, mode, self.varshape)

    # function
    self.integral = odeint(self.derivative, method=method)

  def derivative(self, p, t, V):
    return self.phi * (self.f_p_alpha(V) * (1. - p) - self.f_p_beta(V) * p)

  def update(self, tdi, V):
    self.p.value = self.integral(self.p, tdi['t'], V, tdi['dt'])

  def current(self, V):
    return self.g_max * self.p ** 4 * (self.E - V)

  def reset_state(self, V, batch_size=None):
    alpha = self.f_p_alpha(V)
    beta = self.f_p_beta(V)
    self.p.value = alpha / (alpha + beta)
    if batch_size is not None:
      assert self.p.shape[0] == batch_size

  def f_p_alpha(self, V):
    raise NotImplementedError

  def f_p_beta(self, V):
    raise NotImplementedError


class IKDR_Ba2002(IK_p4_markov):
  r"""The delayed rectifier potassium channel current.

  The potassium current model is adopted from (Bazhenov, et, al. 2002) [1]_.
  It's dynamics is given by:

  .. math::

      \begin{aligned}
      I_{\mathrm{K}} &= g_{\mathrm{max}} * p^4 \\
      \frac{dp}{dt} &= \phi * (\alpha_p (1-p) - \beta_p p) \\
      \alpha_{p} &=\frac{0.032\left(V-V_{sh}-15\right)}{1-\exp \left(-\left(V-V_{sh}-15\right) / 5\right)} \\
      \beta_p &= 0.5 \exp \left(-\left(V-V_{sh}-10\right) / 40\right)
      \end{aligned}

  where :math:`\phi` is a temperature-dependent factor, which is given by
  :math:`\phi=3^{\frac{T-36}{10}}` (:math:`T` is the temperature in Celsius).

  Parameters
  ----------
  size: int, sequence of int
    The object size.
  keep_size: bool
    Whether we use `size` to initialize the variable. Otherwise, variable shape
    will be initialized as `num`.
  g_max : float, JaxArray, ndarray, Initializer, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  E : float, JaxArray, ndarray, Initializer, Callable
    The reversal potential (mV).
  T_base : float, JaxArray, ndarray
    The base of temperature factor.
  T : float, JaxArray, ndarray, Initializer, Callable
    The temperature (Celsius, :math:`^{\circ}C`).
  V_sh : float, JaxArray, ndarray, Initializer, Callable
    The shift of the membrane potential to spike.
  method: str
    The numerical integration method.
  name: str
    The object name.

  References
  ----------
  .. [1] Bazhenov, Maxim, et al. "Model of thalamocortical slow-wave sleep oscillations
         and transitions to activated states." Journal of neuroscience 22.19 (2002): 8691-8704.

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Array, Initializer, Callable] = -90.,
      g_max: Union[float, Array, Initializer, Callable] = 10.,
      V_sh: Union[float, Array, Initializer, Callable] = -50.,
      T_base: Union[float, Array] = 3.,
      T: Union[float, Array] = 36.,
      phi: Optional[Union[float, Array, Initializer, Callable]] = None,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    phi = T_base ** ((T - 36) / 10) if phi is None else phi
    super(IKDR_Ba2002, self).__init__(size,
                                      keep_size=keep_size,
                                      name=name,
                                      method=method,
                                      g_max=g_max,
                                      phi=phi,
                                      E=E,
                                      mode=mode)

    # parameters
    self.T = parameter(T, self.varshape, allow_none=False)
    self.T_base = parameter(T_base, self.varshape, allow_none=False)
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)

  def f_p_alpha(self, V):
    tmp = V - self.V_sh - 15.
    return 0.032 * tmp / (1. - bm.exp(-tmp / 5.))

  def f_p_beta(self, V):
    return 0.5 * bm.exp(-(V - self.V_sh - 10.) / 40.)


class IK_TM1991(IK_p4_markov):
  r"""The potassium channel described by (Traub and Miles, 1991) [1]_.

  The dynamics of this channel is given by:

  .. math::

     \begin{aligned}
      I_{\mathrm{K}} &= g_{\mathrm{max}} * p^4 \\
      \frac{dp}{dt} &= \phi * (\alpha_p (1-p) - \beta_p p) \\
      \alpha_{p} &= 0.032 \frac{(15 - V + V_{sh})}{(\exp((15 - V + V_{sh}) / 5) - 1.)} \\
      \beta_p &= 0.5 * \exp((10 - V + V_{sh}) / 40)
      \end{aligned}

  where :math:`V_{sh}` is the membrane shift (default -63 mV), and
  :math:`\phi` is the temperature-dependent factor (default 1.).

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  g_max : float, JaxArray, ndarray, Initializer, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  E : float, JaxArray, ndarray, Initializer, Callable
    The reversal potential (mV).
  method: str
    The numerical integration method.
  name: str
    The object name.

  References
  ----------
  .. [1] Traub, Roger D., and Richard Miles. Neuronal networks of the hippocampus.
         Vol. 777. Cambridge University Press, 1991.

  See Also
  --------
  INa_TM1991
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Array, Initializer, Callable] = -90.,
      g_max: Union[float, Array, Initializer, Callable] = 10.,
      phi: Union[float, Array, Initializer, Callable] = 1.,
      V_sh: Union[int, float, Array, Initializer, Callable] = -60.,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    super(IK_TM1991, self).__init__(size,
                                    keep_size=keep_size,
                                    name=name,
                                    method=method,
                                    phi=phi,
                                    E=E,
                                    g_max=g_max,
                                    mode=mode)
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)

  def f_p_alpha(self, V):
    c = 15 - V + self.V_sh
    return 0.032 * c / (bm.exp(c / 5) - 1.)

  def f_p_beta(self, V):
    return 0.5 * bm.exp((10 - V + self.V_sh) / 40)


class IK_HH1952(IK_p4_markov):
  r"""The potassium channel described by Hodgkin–Huxley model [1]_.

  The dynamics of this channel is given by:

  .. math::

     \begin{aligned}
      I_{\mathrm{K}} &= g_{\mathrm{max}} * p^4 \\
      \frac{dp}{dt} &= \phi * (\alpha_p (1-p) - \beta_p p) \\
      \alpha_{p} &= \frac{0.01 (V -V_{sh} + 10)}{1-\exp \left(-\left(V-V_{sh}+ 10\right) / 10\right)} \\
      \beta_p &= 0.125 \exp \left(-\left(V-V_{sh}+20\right) / 80\right)
      \end{aligned}

  where :math:`V_{sh}` is the membrane shift (default -45 mV), and
  :math:`\phi` is the temperature-dependent factor (default 1.).

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  g_max : float, JaxArray, ndarray, Initializer, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  E : float, JaxArray, ndarray, Initializer, Callable
    The reversal potential (mV).
  method: str
    The numerical integration method.
  name: str
    The object name.

  References
  ----------
  .. [1] Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description of
         membrane current and its application to conduction and excitation in
         nerve." The Journal of physiology 117.4 (1952): 500.

  See Also
  --------
  INa_HH1952
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Array, Initializer, Callable] = -90.,
      g_max: Union[float, Array, Initializer, Callable] = 10.,
      phi: Union[float, Array, Initializer, Callable] = 1.,
      V_sh: Union[int, float, Array, Initializer, Callable] = -45.,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    super(IK_HH1952, self).__init__(size,
                                    keep_size=keep_size,
                                    name=name,
                                    method=method,
                                    phi=phi,
                                    E=E,
                                    g_max=g_max,
                                    mode=mode)
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)

  def f_p_alpha(self, V):
    temp = V - self.V_sh + 10
    return 0.01 * temp / (1 - bm.exp(-temp / 10))

  def f_p_beta(self, V):
    return 0.125 * bm.exp(-(V - self.V_sh + 20) / 80)


class IKA_p4q_ss(PotassiumChannel):
  r"""The rapidly inactivating Potassium channel of :math:`p^4q`
  current which described with steady-state format.

  This model is developed according to the average behavior of
  rapidly inactivating Potassium channel in Thalamus relay neurons [2]_ [3]_.

  .. math::

     &IA = g_{\mathrm{max}} p^4 q (E-V) \\
     &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
     &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\

  where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  method: str
    The numerical integration method.
  name: str
    The object name.
  g_max : float, JaxArray, ndarray, Initializer, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  E : float, JaxArray, ndarray, Initializer, Callable
    The reversal potential (mV).
  phi_p : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------
  .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
         currents involved in rhythmic oscillations in thalamic relay
         neurons." Journal of neurophysiology 68.4 (1992): 1373-1383.
  .. [3] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
         TEA-sensitive K current in acutely isolated rat thalamic relay
         neurons." Journal of neurophysiology 66.4 (1991): 1316-1328.
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Array, Initializer, Callable] = -90.,
      g_max: Union[float, Array, Initializer, Callable] = 10.,
      phi_p: Union[float, Array, Initializer, Callable] = 1.,
      phi_q: Union[float, Array, Initializer, Callable] = 1.,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    super(IKA_p4q_ss, self).__init__(size,
                                     keep_size=keep_size,
                                     name=name,
                                     mode=mode)

    # parameters
    self.E = parameter(E, self.varshape, allow_none=False)
    self.g_max = parameter(g_max, self.varshape, allow_none=False)
    self.phi_p = parameter(phi_p, self.varshape, allow_none=False)
    self.phi_q = parameter(phi_q, self.varshape, allow_none=False)

    # variables
    self.p = variable(bm.zeros, mode, self.varshape)
    self.q = variable(bm.zeros, mode, self.varshape)

    # function
    self.integral = odeint(JointEq(self.dp, self.dq), method=method)

  def dp(self, p, t, V):
    return self.phi_p * (self.f_p_inf(V) - p) / self.f_p_tau(V)

  def dq(self, q, t, V):
    return self.phi_q * (self.f_q_inf(V) - q) / self.f_q_tau(V)

  def update(self, tdi, V):
    t, dt = tdi['t'], tdi['dt']
    self.p.value, self.q.value = self.integral(self.p.value, self.q.value, t, V, dt)

  def current(self, V):
    return self.g_max * self.p ** 4 * self.q * (self.E - V)

  def reset_state(self, V, batch_size=None):
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


class IKA1_HM1992(IKA_p4q_ss):
  r"""The rapidly inactivating Potassium channel (IA1) model proposed by (Huguenard & McCormick, 1992) [2]_.

  This model is developed according to the average behavior of
  rapidly inactivating Potassium channel in Thalamus relay neurons [2]_ [1]_.

  .. math::

     &IA = g_{\mathrm{max}} p^4 q (E-V) \\
     &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
     &p_{\infty} = \frac{1}{1+ \exp[-(V -V_{sh}+ 60)/8.5]} \\
     &\tau_{p}=\frac{1}{\exp \left(\frac{V -V_{sh}+35.8}{19.7}\right)+ \exp \left(\frac{V -V_{sh}+79.7}{-12.7}\right)}+0.37 \\
     &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\
     &q_{\infty} = \frac{1}{1+ \exp[(V -V_{sh} + 78)/6]} \\
     &\begin{array}{l} \tau_{q} = \frac{1}{\exp((V -V_{sh}+46)/5.) + \exp((V -V_{sh}+238)/-37.5)}  \quad V<(-63+V_{sh})\, mV  \\
          \tau_{q} = 19  \quad V \geq (-63 + V_{sh})\, mV \end{array}

  where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  method: str
    The numerical integration method.
  name: str
    The object name.
  g_max : float, JaxArray, ndarray, Initializer, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  E : float, JaxArray, ndarray, Initializer, Callable
    The reversal potential (mV).
  V_sh : float, Array, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------
  .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
         currents involved in rhythmic oscillations in thalamic relay
         neurons." Journal of neurophysiology 68.4 (1992): 1373-1383.
  .. [1] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
         TEA-sensitive K current in acutely isolated rat thalamic relay
         neurons." Journal of neurophysiology 66.4 (1991): 1316-1328.

  See Also
  --------
  IKA2_HM1992
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Array, Initializer, Callable] = -90.,
      g_max: Union[float, Array, Initializer, Callable] = 30.,
      V_sh: Union[float, Array, Initializer, Callable] = 0.,
      phi_p: Union[float, Array, Initializer, Callable] = 1.,
      phi_q: Union[float, Array, Initializer, Callable] = 1.,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    super(IKA1_HM1992, self).__init__(size,
                                      keep_size=keep_size,
                                      name=name,
                                      method=method,
                                      E=E,
                                      g_max=g_max,
                                      phi_p=phi_p,
                                      phi_q=phi_q,
                                      mode=mode)

    # parameters
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    return 1. / (1. + bm.exp(-(V - self.V_sh + 60.) / 8.5))

  def f_p_tau(self, V):
    return 1. / (bm.exp((V - self.V_sh + 35.8) / 19.7) +
                 bm.exp(-(V - self.V_sh + 79.7) / 12.7)) + 0.37

  def f_q_inf(self, V):
    return 1. / (1. + bm.exp((V - self.V_sh + 78.) / 6.))

  def f_q_tau(self, V):
    return bm.where(V < -63 + self.V_sh,
                    1. / (bm.exp((V - self.V_sh + 46.) / 5.) +
                          bm.exp(-(V - self.V_sh + 238.) / 37.5)),
                    19.)


class IKA2_HM1992(IKA_p4q_ss):
  r"""The rapidly inactivating Potassium channel (IA2) model proposed by (Huguenard & McCormick, 1992) [2]_.

  This model is developed according to the average behavior of
  rapidly inactivating Potassium channel in Thalamus relay neurons [2]_ [1]_.

  .. math::

     &IA = g_{\mathrm{max}} p^4 q (E-V) \\
     &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
     &p_{\infty} = \frac{1}{1+ \exp[-(V -V_{sh}+ 36)/20.]} \\
     &\tau_{p}=\frac{1}{\exp \left(\frac{V -V_{sh}+35.8}{19.7}\right)+ \exp \left(\frac{V -V_{sh}+79.7}{-12.7}\right)}+0.37 \\
     &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\
     &q_{\infty} = \frac{1}{1+ \exp[(V -V_{sh} + 78)/6]} \\
     &\begin{array}{l} \tau_{q} = \frac{1}{\exp((V -V_{sh}+46)/5.) + \exp((V -V_{sh}+238)/-37.5)}  \quad V<(-63+V_{sh})\, mV  \\
          \tau_{q} = 19  \quad V \geq (-63 + V_{sh})\, mV \end{array}

  where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  method: str
    The numerical integration method.
  name: str
    The object name.
  g_max : float, JaxArray, ndarray, Initializer, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  E : float, JaxArray, ndarray, Initializer, Callable
    The reversal potential (mV).
  V_sh : float, Array, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------
  .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
         currents involved in rhythmic oscillations in thalamic relay
         neurons." Journal of neurophysiology 68.4 (1992): 1373-1383.
  .. [1] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
         TEA-sensitive K current in acutely isolated rat thalamic relay
         neurons." Journal of neurophysiology 66.4 (1991): 1316-1328.

  See Also
  --------
  IKA1_HM1992
  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Array, Initializer, Callable] = -90.,
      g_max: Union[float, Array, Initializer, Callable] = 20.,
      V_sh: Union[float, Array, Initializer, Callable] = 0.,
      phi_p: Union[float, Array, Initializer, Callable] = 1.,
      phi_q: Union[float, Array, Initializer, Callable] = 1.,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    super(IKA2_HM1992, self).__init__(size,
                                      keep_size=keep_size,
                                      name=name,
                                      method=method,
                                      E=E,
                                      g_max=g_max,
                                      phi_q=phi_q,
                                      phi_p=phi_p,
                                      mode=mode)

    # parameters
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    return 1. / (1. + bm.exp(-(V - self.V_sh + 36.) / 20.))

  def f_p_tau(self, V):
    return 1. / (bm.exp((V - self.V_sh + 35.8) / 19.7) +
                 bm.exp(-(V - self.V_sh + 79.7) / 12.7)) + 0.37

  def f_q_inf(self, V):
    return 1. / (1. + bm.exp((V - self.V_sh + 78.) / 6.))

  def f_q_tau(self, V):
    return bm.where(V < -63 + self.V_sh,
                    1. / (bm.exp((V - self.V_sh + 46.) / 5.) +
                          bm.exp(-(V - self.V_sh + 238.) / 37.5)),
                    19.)


class IKK2_pq_ss(PotassiumChannel):
  r"""The slowly inactivating Potassium channel of :math:`pq`
  current which described with steady-state format.

  The dynamics of the model is given as [2]_ [3]_.

  .. math::

     &IK2 = g_{\mathrm{max}} p q (E-V) \\
     &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
     &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\

  where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  method: str
    The numerical integration method.
  name: str
    The object name.
  g_max : float, JaxArray, ndarray, Initializer, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  E : float, JaxArray, ndarray, Initializer, Callable
    The reversal potential (mV).
  phi_p : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------
  .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
         currents involved in rhythmic oscillations in thalamic relay
         neurons." Journal of neurophysiology 68.4 (1992): 1373-1383.
  .. [3] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
         TEA-sensitive K current in acutely isolated rat thalamic relay
         neurons." Journal of neurophysiology 66.4 (1991): 1316-1328.

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Array, Initializer, Callable] = -90.,
      g_max: Union[float, Array, Initializer, Callable] = 10.,
      phi_p: Union[float, Array, Initializer, Callable] = 1.,
      phi_q: Union[float, Array, Initializer, Callable] = 1.,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    super(IKK2_pq_ss, self).__init__(size,
                                     keep_size=keep_size,
                                     name=name,
                                     mode=mode)

    # parameters
    self.E = parameter(E, self.varshape, allow_none=False)
    self.g_max = parameter(g_max, self.varshape, allow_none=False)
    self.phi_p = parameter(phi_p, self.varshape, allow_none=False)
    self.phi_q = parameter(phi_q, self.varshape, allow_none=False)

    # variables
    self.p = variable(bm.zeros, mode, self.varshape)
    self.q = variable(bm.zeros, mode, self.varshape)

    # function
    self.integral = odeint(JointEq(self.dp, self.dq), method=method)

  def dp(self, p, t, V):
    return self.phi_p * (self.f_p_inf(V) - p) / self.f_p_tau(V)

  def dq(self, q, t, V):
    return self.phi_q * (self.f_q_inf(V) - q) / self.f_q_tau(V)

  def update(self, tdi, V):
    t, dt = tdi['t'], tdi['dt']
    self.p.value, self.q.value = self.integral(self.p.value, self.q.value, t, V, dt)

  def current(self, V):
    return self.g_max * self.p * self.q * (self.E - V)

  def reset_state(self, V, batch_size=None):
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


class IKK2A_HM1992(IKK2_pq_ss):
  r"""The slowly inactivating Potassium channel (IK2a) model proposed by (Huguenard & McCormick, 1992) [2]_.

  The dynamics of the model is given as [2]_ [3]_.

  .. math::

     &IK2 = g_{\mathrm{max}} p q (E-V) \\
     &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
     &p_{\infty} = \frac{1}{1+ \exp[-(V -V_{sh}+ 43)/17]} \\
     &\tau_{p}=\frac{1}{\exp \left(\frac{V -V_{sh}-81.}{25.6}\right)+
        \exp \left(\frac{V -V_{sh}+132}{-18}\right)}+9.9 \\
     &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\
     &q_{\infty} = \frac{1}{1+ \exp[(V -V_{sh} + 59)/10.6]} \\
     & \tau_{q} = \frac{1}{\exp((V -V_{sh}+1329)/200.) + \exp((V -V_{sh}+130)/-7.1)} + 120 \\

  where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  method: str
    The numerical integration method.
  name: str
    The object name.
  g_max : float, JaxArray, ndarray, Initializer, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  E : float, JaxArray, ndarray, Initializer, Callable
    The reversal potential (mV).
  V_sh : float, Array, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------
  .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
         currents involved in rhythmic oscillations in thalamic relay
         neurons." Journal of neurophysiology 68.4 (1992): 1373-1383.
  .. [3] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
         TEA-sensitive K current in acutely isolated rat thalamic relay
         neurons." Journal of neurophysiology 66.4 (1991): 1316-1328.

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Array, Initializer, Callable] = -90.,
      g_max: Union[float, Array, Initializer, Callable] = 10.,
      V_sh: Union[float, Array, Initializer, Callable] = 0.,
      phi_p: Union[float, Array, Initializer, Callable] = 1.,
      phi_q: Union[float, Array, Initializer, Callable] = 1.,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    super(IKK2A_HM1992, self).__init__(size,
                                       keep_size=keep_size,
                                       name=name,
                                       method=method,
                                       phi_p=phi_p,
                                       phi_q=phi_q,
                                       g_max=g_max,
                                       E=E,
                                       mode=mode)

    # parameters
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    raise 1. / (1. + bm.exp(-(V - self.V_sh + 43.) / 17.))

  def f_p_tau(self, V):
    return 1. / (bm.exp((V - self.V_sh - 81.) / 25.6) +
                 bm.exp(-(V - self.V_sh + 132) / 18.)) + 9.9

  def f_q_inf(self, V):
    raise 1. / (1. + bm.exp((V - self.V_sh + 58.) / 10.6))

  def f_q_tau(self, V):
    raise 1. / (bm.exp((V - self.V_sh - 1329.) / 200.) +
                bm.exp(-(V - self.V_sh + 130.) / 7.1))


class IKK2B_HM1992(IKK2_pq_ss):
  r"""The slowly inactivating Potassium channel (IK2b) model proposed by (Huguenard & McCormick, 1992) [2]_.

  The dynamics of the model is given as [2]_ [3]_.

  .. math::

     &IK2 = g_{\mathrm{max}} p q (E-V) \\
     &\frac{dp}{dt} = \phi_p \frac{p_{\infty} - p}{\tau_p} \\
     &p_{\infty} = \frac{1}{1+ \exp[-(V -V_{sh}+ 43)/17]} \\
     &\tau_{p}=\frac{1}{\exp \left(\frac{V -V_{sh}-81.}{25.6}\right)+
     \exp \left(\frac{V -V_{sh}+132}{-18}\right)}+9.9 \\
     &\frac{dq}{dt} = \phi_q \frac{q_{\infty} - q}{\tau_q} \\
     &q_{\infty} = \frac{1}{1+ \exp[(V -V_{sh} + 59)/10.6]} \\
     &\begin{array}{l} \tau_{q} = \frac{1}{\exp((V -V_{sh}+1329)/200.) +
                      \exp((V -V_{sh}+130)/-7.1)} + 120 \quad V<(-70+V_{sh})\, mV  \\
          \tau_{q} = 8.9  \quad V \geq (-70 + V_{sh})\, mV \end{array}

  where :math:`\phi_p` and :math:`\phi_q` are the temperature dependent factors (default 1.).

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  method: str
    The numerical integration method.
  name: str
    The object name.
  g_max : float, JaxArray, ndarray, Initializer, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  E : float, JaxArray, ndarray, Initializer, Callable
    The reversal potential (mV).
  V_sh : float, Array, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`p`.
  phi_q : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`q`.

  References
  ----------
  .. [2] Huguenard, John R., and David A. McCormick. "Simulation of the
         currents involved in rhythmic oscillations in thalamic relay
         neurons." Journal of neurophysiology 68.4 (1992): 1373-1383.
  .. [3] Huguenard, J. R., and D. A. Prince. "Slow inactivation of a
         TEA-sensitive K current in acutely isolated rat thalamic relay
         neurons." Journal of neurophysiology 66.4 (1991): 1316-1328.

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Array, Initializer, Callable] = -90.,
      g_max: Union[float, Array, Initializer, Callable] = 10.,
      V_sh: Union[float, Array, Initializer, Callable] = 0.,
      phi_p: Union[float, Array, Initializer, Callable] = 1.,
      phi_q: Union[float, Array, Initializer, Callable] = 1.,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    super(IKK2B_HM1992, self).__init__(size,
                                       keep_size=keep_size,
                                       name=name,
                                       method=method,
                                       phi_p=phi_p,
                                       phi_q=phi_q,
                                       g_max=g_max,
                                       E=E,
                                       mode=mode)

    # parameters
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)

  def f_p_inf(self, V):
    raise 1. / (1. + bm.exp(-(V - self.V_sh + 43.) / 17.))

  def f_p_tau(self, V):
    return 1. / (bm.exp((V - self.V_sh - 81.) / 25.6) +
                 bm.exp(-(V - self.V_sh + 132) / 18.)) + 9.9

  def f_q_inf(self, V):
    raise 1. / (1. + bm.exp((V - self.V_sh + 58.) / 10.6))

  def f_q_tau(self, V):
    raise bm.where(V < -70 + self.V_sh,
                   1. / (bm.exp((V - self.V_sh - 1329.) / 200.) +
                         bm.exp(-(V - self.V_sh + 130.) / 7.1)),
                   8.9)


class IKNI_Ya1989(PotassiumChannel):
  r"""A slow non-inactivating K+ current described by Yamada et al. (1989) [1]_.

  This slow potassium current can effectively account for spike-frequency adaptation.

  .. math::

    \begin{aligned}
    &I_{M}=\bar{g}_{M} p\left(V-E_{K}\right) \\
    &\frac{\mathrm{d} p}{\mathrm{~d} t}=\left(p_{\infty}(V)-p\right) / \tau_{p}(V) \\
    &p_{\infty}(V)=\frac{1}{1+\exp [-(V-V_{sh}+35) / 10]} \\
    &\tau_{p}(V)=\frac{\tau_{\max }}{3.3 \exp [(V-V_{sh}+35) / 20]+\exp [-(V-V_{sh}+35) / 20]}
    \end{aligned}

  where :math:`\bar{g}_{M}` was :math:`0.004 \mathrm{mS} / \mathrm{cm}^{2}` and
  :math:`\tau_{\max }=4 \mathrm{~s}`, unless stated otherwise.

  Parameters
  ----------
  size: int, sequence of int
    The geometry size.
  method: str
    The numerical integration method.
  name: str
    The object name.
  g_max : float, JaxArray, ndarray, Initializer, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  E : float, JaxArray, ndarray, Initializer, Callable
    The reversal potential (mV).
  V_sh : float, Array, Callable, Initializer
    The membrane potential shift.
  phi_p : optional, float, Array, Callable, Initializer
    The temperature factor for channel :math:`p`.
  tau_max: float, Array, Callable, Initializer
    The :math:`tau_{\max}` parameter.

  References
  ----------
  .. [1] Yamada, Walter M. "Multiple channels and calcium dynamics." Methods in neuronal modeling (1989): 97-133.

  """

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Array, Initializer, Callable] = -90.,
      g_max: Union[float, Array, Initializer, Callable] = 0.004,
      phi_p: Union[float, Array, Initializer, Callable] = 1.,
      phi_q: Union[float, Array, Initializer, Callable] = 1.,
      tau_max: Union[float, Array, Initializer, Callable] = 4e3,
      V_sh: Union[float, Array, Initializer, Callable] = 0.,
      method: str = 'exp_auto',
      name: str = None,
      mode: Mode = normal,
  ):
    super(IKNI_Ya1989, self).__init__(size,
                                      keep_size=keep_size,
                                      name=name,
                                      mode=mode)

    # parameters
    self.E = parameter(E, self.varshape, allow_none=False)
    self.g_max = parameter(g_max, self.varshape, allow_none=False)
    self.tau_max = parameter(tau_max, self.varshape, allow_none=False)
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)
    self.phi_p = parameter(phi_p, self.varshape, allow_none=False)
    self.phi_q = parameter(phi_q, self.varshape, allow_none=False)

    # variables
    self.p = variable(bm.zeros, mode, self.varshape)

    # function
    self.integral = odeint(self.dp, method=method)

  def dp(self, p, t, V):
    return self.phi_p * (self.f_p_inf(V) - p) / self.f_p_tau(V)

  def update(self, tdi, V):
    t, dt = tdi['t'], tdi['dt']
    self.p.value = self.integral(self.p.value, t, V, dt)

  def current(self, V):
    return self.g_max * self.p * (self.E - V)

  def reset_state(self, V, batch_size=None):
    self.p.value = self.f_p_inf(V)
    if batch_size is not None:
      assert self.p.shape[0] == batch_size

  def f_p_inf(self, V):
    raise 1. / (1. + bm.exp(-(V - self.V_sh + 35.) / 10.))

  def f_p_tau(self, V):
    temp = V - self.V_sh + 35.
    raise self.tau_max / (3.3 * bm.exp(temp / 20.) + bm.exp(-temp / 20.))
