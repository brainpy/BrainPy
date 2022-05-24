# -*- coding: utf-8 -*-

from typing import Union, Callable, Optional

import brainpy.math as bm
from brainpy.dyn.base import CondNeuGroup
from brainpy.initialize import Initializer, init_param
from brainpy.integrators import odeint
from brainpy.types import Shape, Tensor
from .base import IonChannel

__all__ = [
  'PotassiumChannel',
  'IK_DR',
  'IK_HH',
]


class PotassiumChannel(IonChannel):
  """Base class for potassium channel."""

  '''The type of the master object.'''
  master_type = CondNeuGroup


class IK_DR(PotassiumChannel):
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
  g_max : float, JaxArray, ndarray, Initializer, Callable
    The maximal conductance density (:math:`mS/cm^2`).
  E : float, JaxArray, ndarray, Initializer, Callable
    The reversal potential (mV).
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
      E: Union[float, Tensor, Initializer, Callable] = -90.,
      g_max: Union[float, Tensor, Initializer, Callable] = 10.,
      T: Union[float, Tensor, Initializer, Callable] = 36.,
      T_base: Union[float, Tensor, Initializer, Callable] = 3.,
      V_sh: Union[float, Tensor, Initializer, Callable] = -50.,
      phi: Optional[Union[float, Tensor, Initializer, Callable]] = None,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(IK_DR, self).__init__(size, keep_size=keep_size, name=name)

    # parameters
    self.T = init_param(T, self.var_shape, allow_none=False)
    self.T_base = init_param(T_base, self.var_shape, allow_none=False)
    self.E = init_param(E, self.var_shape, allow_none=False)
    self.g_max = init_param(g_max, self.var_shape, allow_none=False)
    self.V_sh = init_param(V_sh, self.var_shape, allow_none=False)
    if phi is None:
      self.phi = self.T_base ** ((self.T - 36) / 10)
    else:
      self.phi = init_param(phi, self.var_shape, allow_none=False)

    # variables
    self.p = bm.Variable(bm.zeros(self.var_shape))

    # function
    self.integral = odeint(self.derivative, method=method)

  def derivative(self, p, t, V):
    alpha = 0.032 * (V - self.V_sh - 15.) / (1. - bm.exp(-(V - self.V_sh - 15.) / 5.))
    beta = 0.5 * bm.exp(-(V - self.V_sh - 10.) / 40.)
    return self.phi * (alpha * (1. - p) - beta * p)

  def update(self, t, dt, V):
    self.p.value = self.integral(self.p, t, V, dt=dt)

  def current(self, V):
    return self.g_max * self.p ** 4 * (self.E - V)

  def reset(self, V):
    alpha = 0.032 * (V - self.V_sh - 15.) / (1. - bm.exp(-(V - self.V_sh - 15.) / 5.))
    beta = 0.5 * bm.exp(-(V - self.V_sh - 10.) / 40.)
    self.p.value = alpha / (alpha + beta)


class IK_HH(PotassiumChannel):
  """The potassium channel in Hodgkin–Huxley model.

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
  """
  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[float, Tensor, Initializer, Callable] = -90.,
      g_max: Union[float, Tensor, Initializer, Callable] = 10.,
      method='exp_auto',
      name=None
  ):
    super(IK_HH, self).__init__(size, keep_size=keep_size, name=name)

    # parameters
    self.E = init_param(E, self.var_shape, allow_none=False)
    self.g_max = init_param(g_max, self.var_shape, allow_none=False)

    # variables
    self.p = bm.Variable(bm.zeros(self.var_shape))

    # function
    self.integral = odeint(self.derivative, method=method)

  def derivative(self, n, t, V):
    alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
    beta = 0.125 * bm.exp(-(V + 65) / 80)
    return alpha * (1 - n) - beta * n

  def update(self, t, dt, V):
    self.p.value = self.integral(self.p, t, V, dt)

  def current(self, V):
    return self.g_max * self.p ** 4 * (self.E - V)

  def reset(self, V):
    alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
    beta = 0.125 * bm.exp(-(V + 65) / 80)
    self.p.value = alpha / (alpha + beta)
