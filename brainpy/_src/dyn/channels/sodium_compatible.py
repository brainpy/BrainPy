# -*- coding: utf-8 -*-

"""
This module implements voltage-dependent sodium channels.

"""

from typing import Union, Callable, Sequence

import brainpy.math as bm
from brainpy._src.context import share
from brainpy._src.dyn.neurons.hh import HHTypedNeuron
from brainpy._src.initialize import Initializer, parameter, variable
from brainpy._src.integrators import odeint, JointEq
from brainpy.types import ArrayType
from .base import IonChannel

__all__ = [
  'INa_Ba2002',
  'INa_TM1991',
  'INa_HH1952',
]


class _INa_p3q_markov(IonChannel):
  r"""The sodium current model of :math:`p^3q` current which described with first-order Markov chain.

  The general model can be used to model the dynamics with:

  .. math::

    \begin{aligned}
    I_{\mathrm{Na}} &= g_{\mathrm{max}} * p^3 * q \\
    \frac{dp}{dt} &= \phi ( \alpha_p (1-p) - \beta_p p) \\
    \frac{dq}{dt} & = \phi ( \alpha_q (1-h) - \beta_q h) \\
    \end{aligned}

  where :math:`\phi` is a temperature-dependent factor.

  Parameters
  ----------
  g_max : float, ArrayType, Callable, Initializer
    The maximal conductance density (:math:`mS/cm^2`).
  E : float, ArrayType, Callable, Initializer
    The reversal potential (mV).
  phi : float, ArrayType, Callable, Initializer
    The temperature-dependent factor.
  method: str
    The numerical method
  name: str
    The name of the object.

  """
  master_type = HHTypedNeuron

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      E: Union[int, float, ArrayType, Initializer, Callable] = None,
      g_max: Union[int, float, ArrayType, Initializer, Callable] = 90.,
      phi: Union[int, float, ArrayType, Initializer, Callable] = 1.,
      method: str = 'exp_auto',
      name: str = None,
      mode: bm.Mode = None,
  ):
    super().__init__(size=size,
                                          keep_size=keep_size,
                                          name=name,
                                          mode=mode)

    # parameters
    self.E = parameter(E, self.varshape, allow_none=False)
    self.phi = parameter(phi, self.varshape, allow_none=False)
    self.g_max = parameter(g_max, self.varshape, allow_none=False)

    # variables
    self.p = variable(bm.zeros, self.mode, self.varshape)
    self.q = variable(bm.zeros, self.mode, self.varshape)

    # function
    self.integral = odeint(JointEq([self.dp, self.dq]), method=method)

  def reset_state(self, V, batch_size=None):
    alpha = self.f_p_alpha(V)
    beta = self.f_p_beta(V)
    self.p.value = alpha / (alpha + beta)
    alpha = self.f_q_alpha(V)
    beta = self.f_q_beta(V)
    self.q.value = alpha / (alpha + beta)
    if isinstance(batch_size, int):
      assert self.p.shape[0] == batch_size
      assert self.q.shape[0] == batch_size

  def dp(self, p, t, V):
    return self.phi * (self.f_p_alpha(V) * (1. - p) - self.f_p_beta(V) * p)

  def dq(self, q, t, V):
    return self.phi * (self.f_q_alpha(V) * (1. - q) - self.f_q_beta(V) * q)

  def update(self, V):
    p, q = self.integral(self.p, self.q, share['t'], V, share['dt'])
    self.p.value, self.q.value = p, q

  def current(self, V):
    return self.g_max * self.p ** 3 * self.q * (self.E - V)

  def f_p_alpha(self, V):
    raise NotImplementedError

  def f_p_beta(self, V):
    raise NotImplementedError

  def f_q_alpha(self, V):
    raise NotImplementedError

  def f_q_beta(self, V):
    raise NotImplementedError


class INa_Ba2002(_INa_p3q_markov):
  r"""The sodium current model.

  The sodium current model is adopted from (Bazhenov, et, al. 2002) [1]_.
  It's dynamics is given by:

  .. math::

    \begin{aligned}
    I_{\mathrm{Na}} &= g_{\mathrm{max}} * p^3 * q \\
    \frac{dp}{dt} &= \phi ( \alpha_p (1-p) - \beta_p p) \\
    \alpha_{p} &=\frac{0.32\left(V-V_{sh}-13\right)}{1-\exp \left(-\left(V-V_{sh}-13\right) / 4\right)} \\
    \beta_{p} &=\frac{-0.28\left(V-V_{sh}-40\right)}{1-\exp \left(\left(V-V_{sh}-40\right) / 5\right)} \\
    \frac{dq}{dt} & = \phi ( \alpha_q (1-h) - \beta_q h) \\
    \alpha_q &=0.128 \exp \left(-\left(V-V_{sh}-17\right) / 18\right) \\
    \beta_q &= \frac{4}{1+\exp \left(-\left(V-V_{sh}-40\right) / 5\right)}
    \end{aligned}

  where :math:`\phi` is a temperature-dependent factor, which is given by
  :math:`\phi=3^{\frac{T-36}{10}}` (:math:`T` is the temperature in Celsius).

  Parameters
  ----------
  g_max : float, ArrayType, Callable, Initializer
    The maximal conductance density (:math:`mS/cm^2`).
  E : float, ArrayType, Callable, Initializer
    The reversal potential (mV).
  T : float, ArrayType
    The temperature (Celsius, :math:`^{\circ}C`).
  V_sh : float, ArrayType, Callable, Initializer
    The shift of the membrane potential to spike.

  References
  ----------

  .. [1] Bazhenov, Maxim, et al. "Model of thalamocortical slow-wave sleep oscillations
         and transitions to activated states." Journal of neuroscience 22.19 (2002): 8691-8704.

  See Also
  --------
  INa_TM1991
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      T: Union[int, float, ArrayType] = 36.,
      E: Union[int, float, ArrayType, Initializer, Callable] = 50.,
      g_max: Union[int, float, ArrayType, Initializer, Callable] = 90.,
      V_sh: Union[int, float, ArrayType, Initializer, Callable] = -50.,
      method: str = 'exp_auto',
      name: str = None,
      mode: bm.Mode = None,
  ):
    super().__init__(size,
                                     keep_size=keep_size,
                                     name=name,
                                     method=method,
                                     phi=3 ** ((T - 36) / 10),
                                     g_max=g_max,
                                     E=E,
                                     mode=mode)
    self.T = parameter(T, self.varshape, allow_none=False)
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)

  def f_p_alpha(self, V):
    temp = V - self.V_sh - 13.
    return 0.32 * temp / (1. - bm.exp(-temp / 4.))

  def f_p_beta(self, V):
    temp = V - self.V_sh - 40.
    return -0.28 * temp / (1. - bm.exp(temp / 5.))

  def f_q_alpha(self, V):
    return 0.128 * bm.exp(-(V - self.V_sh - 17.) / 18.)

  def f_q_beta(self, V):
    return 4. / (1. + bm.exp(-(V - self.V_sh - 40.) / 5.))


class INa_TM1991(_INa_p3q_markov):
  r"""The sodium current model described by (Traub and Miles, 1991) [1]_.

  The dynamics of this sodium current model is given by:

  .. math::

     \begin{split}
     \begin{aligned}
        I_{\mathrm{Na}} &= g_{\mathrm{max}} m^3 h \\
        \frac {dm} {dt} &= \phi(\alpha_m (1-x)  - \beta_m) \\
        &\alpha_m(V) = 0.32 \frac{(13 - V + V_{sh})}{\exp((13 - V +V_{sh}) / 4) - 1.}  \\
        &\beta_m(V) = 0.28 \frac{(V - V_{sh} - 40)}{(\exp((V - V_{sh} - 40) / 5) - 1)}  \\
        \frac {dh} {dt} &= \phi(\alpha_h (1-x)  - \beta_h) \\
        &\alpha_h(V) = 0.128 * \exp((17 - V + V_{sh}) / 18)  \\
        &\beta_h(V) = 4. / (1 + \exp(-(V - V_{sh} - 40) / 5)) \\
     \end{aligned}
     \end{split}

  where :math:`V_{sh}` is the membrane shift (default -63 mV), and
  :math:`\phi` is the temperature-dependent factor (default 1.).

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
    The maximal conductance density (:math:`mS/cm^2`).
  E : float, ArrayType, Callable, Initializer
    The reversal potential (mV).
  V_sh: float, ArrayType, Callable, Initializer
    The membrane shift.

  References
  ----------
  .. [1] Traub, Roger D., and Richard Miles. Neuronal networks of the hippocampus.
         Vol. 777. Cambridge University Press, 1991.

  See Also
  --------
  INa_Ba2002
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      E: Union[int, float, ArrayType, Initializer, Callable] = 50.,
      g_max: Union[int, float, ArrayType, Initializer, Callable] = 120.,
      phi: Union[int, float, ArrayType, Initializer, Callable] = 1.,
      V_sh: Union[int, float, ArrayType, Initializer, Callable] = -63.,
      method: str = 'exp_auto',
      name: str = None,
      mode: bm.Mode = None,
  ):
    super().__init__(size,
                                     keep_size=keep_size,
                                     name=name,
                                     method=method,
                                     E=E,
                                     phi=phi,
                                     g_max=g_max,
                                     mode=mode)
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)

  def f_p_alpha(self, V):
    temp = 13 - V + self.V_sh
    return 0.32 * temp / (bm.exp(temp / 4) - 1.)

  def f_p_beta(self, V):
    temp = V - self.V_sh - 40
    return 0.28 * temp / (bm.exp(temp / 5) - 1)

  def f_q_alpha(self, V):
    return 0.128 * bm.exp((17 - V + self.V_sh) / 18)

  def f_q_beta(self, V):
    return 4. / (1 + bm.exp(-(V - self.V_sh - 40) / 5))


class INa_HH1952(_INa_p3q_markov):
  r"""The sodium current model described by Hodgkin–Huxley model [1]_.

  The dynamics of this sodium current model is given by:

  .. math::

     \begin{split}
     \begin{aligned}
        I_{\mathrm{Na}} &= g_{\mathrm{max}} m^3 h \\
        \frac {dm} {dt} &= \phi (\alpha_m (1-x)  - \beta_m) \\
        &\alpha_m(V) = \frac {0.1(V-V_{sh}-5)}{1-\exp(\frac{-(V -V_{sh} -5)} {10})}  \\
        &\beta_m(V) = 4.0 \exp(\frac{-(V -V_{sh}+ 20)} {18})  \\
        \frac {dh} {dt} &= \phi (\alpha_h (1-x)  - \beta_h) \\
        &\alpha_h(V) = 0.07 \exp(\frac{-(V-V_{sh}+20)}{20})  \\
        &\beta_h(V) = \frac 1 {1 + \exp(\frac{-(V -V_{sh}-10)} {10})} \\
     \end{aligned}
     \end{split}

  where :math:`V_{sh}` is the membrane shift (default -45 mV), and
  :math:`\phi` is the temperature-dependent factor (default 1.).

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
    The maximal conductance density (:math:`mS/cm^2`).
  E : float, ArrayType, Callable, Initializer
    The reversal potential (mV).
  V_sh: float, ArrayType, Callable, Initializer
    The membrane shift.

  References
  ----------
  .. [1] Hodgkin, Alan L., and Andrew F. Huxley. "A quantitative description of
         membrane current and its application to conduction and excitation in
         nerve." The Journal of physiology 117.4 (1952): 500.

  See Also
  --------
  IK_HH1952
  """

  def __init__(
      self,
      size: Union[int, Sequence[int]],
      keep_size: bool = False,
      E: Union[int, float, ArrayType, Initializer, Callable] = 50.,
      g_max: Union[int, float, ArrayType, Initializer, Callable] = 120.,
      phi: Union[int, float, ArrayType, Initializer, Callable] = 1.,
      V_sh: Union[int, float, ArrayType, Initializer, Callable] = -45.,
      method: str = 'exp_auto',
      name: str = None,
      mode: bm.Mode = None,
  ):
    super().__init__(size,
                                     keep_size=keep_size,
                                     name=name,
                                     method=method,
                                     E=E,
                                     phi=phi,
                                     g_max=g_max,
                                     mode=mode)
    self.V_sh = parameter(V_sh, self.varshape, allow_none=False)

  def f_p_alpha(self, V):
    temp = V - self.V_sh - 5
    return 0.1 * temp / (1 - bm.exp(-temp / 10))

  def f_p_beta(self, V):
    return 4.0 * bm.exp(-(V - self.V_sh + 20) / 18)

  def f_q_alpha(self, V):
    return 0.07 * bm.exp(-(V - self.V_sh + 20) / 20.)

  def f_q_beta(self, V):
    return 1 / (1 + bm.exp(-(V - self.V_sh - 10) / 10))
