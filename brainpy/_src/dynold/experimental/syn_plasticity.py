# -*- coding: utf-8 -*-

from typing import Union

import jax.numpy as jnp

from brainpy import math as bm, tools
from brainpy._src.context import share
from brainpy._src.dynold.experimental.base import SynSTPNS
from brainpy._src.initialize import variable_, OneInit, parameter
from brainpy._src.integrators import odeint, JointEq
from brainpy.types import ArrayType, Shape

__all__ = [
  'STD',
  'STP',
]


class STD(SynSTPNS):
  r"""Synaptic output with short-term depression.

  This model filters the synaptic current by the following equation:

  .. math::

     I_{syn}^+(t) = I_{syn}^-(t) * x

  where :math:`x` is the normalized variable between 0 and 1, and
  :math:`I_{syn}^-(t)` and :math:`I_{syn}^+(t)` are the synaptic currents before
  and after STD filtering.

  Moreover, :math:`x` is updated according to the dynamics of:

  .. math::

     \frac{dx}{dt} = \frac{1-x}{\tau} - U * x * \delta(t-t_{spike})

  where :math:`U` is the fraction of resources used per action potential,
  :math:`\tau` is the time constant of recovery of the synaptic vesicles.

  Parameters
  ----------
  tau: float
    The time constant of recovery of the synaptic vesicles.
  U: float
    The fraction of resources used per action potential.

  See Also
  --------
  STP
  """

  def __init__(
      self,
      pre_size: Shape,
      tau: float = 200.,
      U: float = 0.07,
      method: str = 'exp_auto',
      name: str = None
  ):
    super().__init__(name=name)

    # parameters
    self.pre_size = tools.to_size(pre_size)
    self.num = tools.size2num(self.pre_size)
    self.U = parameter(U, self.num)
    self.tau = parameter(tau, self.num)
    self.method = method

    # integral function
    self.integral = odeint(lambda x, t: (1 - x) / self.tau, method=self.method)

    # variables
    self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    self.x = variable_(jnp.ones, self.num, batch_size)

  def update(self, pre_spike):
    x = self.integral(self.x.value, share.load('t'), share.load('dt'))
    self.x.value = bm.where(pre_spike, x - self.U * self.x, x)
    return self.x.value


class STP(SynSTPNS):
  r"""Synaptic output with short-term plasticity.

  This model filters the synaptic currents according to two variables: :math:`u` and :math:`x`.

  .. math::

     I_{syn}^+(t) = I_{syn}^-(t) * x * u

  where :math:`I_{syn}^-(t)` and :math:`I_{syn}^+(t)` are the synaptic currents before
  and after STP filtering, :math:`x` denotes the fraction of resources that remain available
  after neurotransmitter depletion, and :math:`u` represents the fraction of available
  resources ready for use (release probability).

  The dynamics of :math:`u` and :math:`x` are governed by

  .. math::

     \begin{aligned}
    \frac{du}{dt} & = & -\frac{u}{\tau_f}+U(1-u^-)\delta(t-t_{sp}), \\
    \frac{dx}{dt} & = & \frac{1-x}{\tau_d}-u^+x^-\delta(t-t_{sp}), \\
    \tag{1}\end{aligned}

  where :math:`t_{sp}` denotes the spike time and :math:`U` is the increment
  of :math:`u` produced by a spike. :math:`u^-, x^-` are the corresponding
  variables just before the arrival of the spike, and :math:`u^+`
  refers to the moment just after the spike.

  Parameters
  ----------
  tau_f: float
    The time constant of short-term facilitation.
  tau_d: float
    The time constant of short-term depression.
  U: float
    The fraction of resources used per action potential.
  method: str
    The numerical integral method.

  See Also
  --------
  STD
  """

  def __init__(
      self,
      pre_size: Shape,
      U: Union[float, ArrayType] = 0.15,
      tau_f: Union[float, ArrayType] = 1500.,
      tau_d: Union[float, ArrayType] = 200.,
      method: str = 'exp_auto',
      name: str = None
  ):
    super().__init__(name=name)

    # parameters
    self.pre_size = tools.to_size(pre_size)
    self.num = tools.size2num(self.pre_size)
    self.tau_f = parameter(tau_f, self.num)
    self.tau_d = parameter(tau_d, self.num)
    self.U = parameter(U, self.num)
    self.method = method

    # integral function
    self.integral = odeint(JointEq([self.du, self.dx]), method=self.method)

    # variables
    self.reset_state(self.mode)

  def reset_state(self, batch_size=None):
    self.x = variable_(jnp.ones, batch_size, self.num)
    self.u = variable_(OneInit(self.U), batch_size, self.num)

  du = lambda self, u, t: self.U - u / self.tau_f
  dx = lambda self, x, t: (1 - x) / self.tau_d

  def update(self, pre_spike):
    u, x = self.integral(self.u.value, self.x.value, share.load('t'), bm.get_dt())
    u = bm.where(pre_spike, u + self.U * (1 - self.u), u)
    x = bm.where(pre_spike, x - u * self.x, x)
    self.x.value = x
    self.u.value = u
    return self.x.value * self.u.value
