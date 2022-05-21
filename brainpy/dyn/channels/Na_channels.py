# -*- coding: utf-8 -*-

from typing import Union, Callable

import brainpy.math as bm
from brainpy.dyn.base import CondNeuGroup
from brainpy.initialize import Initializer, init_param
from brainpy.integrators import odeint, JointEq
from brainpy.types import Tensor, Shape
from .base import IonChannel

__all__ = [
  'SodiumChannel',
  'INa',
  'INa_HH',
]


class SodiumChannel(IonChannel):
  """Base class for sodium channel."""
  master_type = CondNeuGroup


class INa(SodiumChannel):
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

  **Model Examples**

  - `(Brette, et, al., 2007) COBAHH <../../examples/ei_nets/Brette_2007_COBAHH.ipynb>`_

  Parameters
  ----------
  g_max : float
    The maximal conductance density (:math:`mS/cm^2`).
  E : float
    The reversal potential (mV).
  T : float
    The temperature (Celsius, :math:`^{\circ}C`).
  V_sh : float
    The shift of the membrane potential to spike.

  References
  ----------

  .. [1] Bazhenov, Maxim, et al. "Model of thalamocortical slow-wave sleep oscillations
         and transitions to activated states." Journal of neuroscience 22.19 (2002): 8691-8704.

  """
  master_type = CondNeuGroup

  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[int, float, Tensor, Initializer, Callable] = 50.,
      g_max: Union[int, float, Tensor, Initializer, Callable] = 90.,
      T: Union[int, float, Tensor, Initializer, Callable] = 36.,
      V_sh: Union[int, float, Tensor, Initializer, Callable] = -50.,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(INa, self).__init__(size, keep_size=keep_size, name=name)

    # parameters
    self.T = init_param(T, self.var_shape, allow_none=False)
    self.E = init_param(E, self.var_shape, allow_none=False)
    self.V_sh = init_param(V_sh, self.var_shape, allow_none=False)
    self.g_max = init_param(g_max, self.var_shape, allow_none=False)
    self.phi = 3 ** ((self.T - 36) / 10)

    # variables
    self.p = bm.Variable(bm.zeros(self.var_shape))
    self.q = bm.Variable(bm.zeros(self.var_shape))

    # function
    self.integral = odeint(JointEq([self.dp, self.dq]), method=method)

  def reset(self, V):
    alpha = 0.32 * (V - self.V_sh - 13.) / (1. - bm.exp(-(V - self.V_sh - 13.) / 4.))
    beta = -0.28 * (V - self.V_sh - 40.) / (1. - bm.exp((V - self.V_sh - 40.) / 5.))
    self.p.value = alpha / (alpha + beta)
    alpha = 0.128 * bm.exp(-(V - self.V_sh - 17.) / 18.)
    beta = 4. / (1. + bm.exp(-(V - self.V_sh - 40.) / 5.))
    self.q.value = alpha / (alpha + beta)

  def dp(self, p, t, V):
    alpha_p = 0.32 * (V - self.V_sh - 13.) / (1. - bm.exp(-(V - self.V_sh - 13.) / 4.))
    beta_p = -0.28 * (V - self.V_sh - 40.) / (1. - bm.exp((V - self.V_sh - 40.) / 5.))
    return self.phi * (alpha_p * (1. - p) - beta_p * p)

  def dq(self, q, t, V):
    alpha_q = 0.128 * bm.exp(-(V - self.V_sh - 17.) / 18.)
    beta_q = 4. / (1. + bm.exp(-(V - self.V_sh - 40.) / 5.))
    return self.phi * (alpha_q * (1. - q) - beta_q * q)

  def update(self, t, dt, V):
    p, q = self.integral(self.p, self.q, t, V, dt)
    self.p.value, self.q.value = p, q

  def current(self, V):
    return self.g_max * self.p ** 3 * self.q * (self.E - V)


class INa_HH(SodiumChannel):
  def __init__(
      self,
      size: Shape,
      keep_size: bool = False,
      E: Union[int, float, Tensor, Initializer, Callable] = 50.,
      g_max: Union[int, float, Tensor, Initializer, Callable] = 120.,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(INa_HH, self).__init__(size, keep_size=keep_size, name=name)

    # parameters
    self.E = init_param(E, self.var_shape, allow_none=False)
    self.g_max = init_param(g_max, self.var_shape, allow_none=False)

    # variables
    self.p = bm.Variable(bm.zeros(self.var_shape))
    self.q = bm.Variable(bm.zeros(self.var_shape))

    # function
    self.integral = odeint(JointEq([self.dp, self.dq]), method=method)

  def dp(self, m, t, V):
    alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
    beta = 4.0 * bm.exp(-(V + 65) / 18)
    return alpha * (1 - m) - beta * m

  def dq(self, h, t, V):
    alpha = 0.07 * bm.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bm.exp(-(V + 35) / 10))
    return alpha * (1 - h) - beta * h

  def update(self, t, dt, V):
    self.p.value, self.q.value = self.integral(self.p, self.q, t, V, dt)

  def current(self, V):
    g = self.g_max * self.p ** 3 * self.q
    return g * (self.E - V)

  def reset(self, V):
    alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
    beta = 4.0 * bm.exp(-(V + 65) / 18)
    self.p.value = alpha / (alpha + beta)
    alpha = 0.07 * bm.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bm.exp(-(V + 35) / 10))
    self.q.value = alpha / (alpha + beta)
