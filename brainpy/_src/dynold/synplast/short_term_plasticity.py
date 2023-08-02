# -*- coding: utf-8 -*-

from typing import Union

import jax.numpy as jnp

from brainpy._src.context import share
from brainpy._src.dynold.synapses.base import _SynSTP
from brainpy._src.initialize import variable
from brainpy._src.integrators import odeint, JointEq
from brainpy.check import is_float
from brainpy.types import ArrayType

__all__ = [
  'STD',
  'STP',
]


class STD(_SynSTP):
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
      tau: float = 200.,
      U: float = 0.07,
      method: str = 'exp_auto',
      name: str = None
  ):
    super().__init__(name=name)

    # parameters
    is_float(tau, 'tau', min_bound=0, )
    is_float(U, 'U', min_bound=0, )
    self.tau = tau
    self.U = U
    self.method = method

    # integral function
    self.integral = odeint(lambda x, t: (1 - x) / self.tau, method=self.method)

  def clone(self):
    return STD(tau=self.tau, U=self.U, method=self.method)

  def register_master(self, master):
    super().register_master(master)
    self.x = variable(jnp.ones, self.master.mode, self.master.pre.num)

  def reset_state(self, batch_size=None):
    self.x.value = variable(jnp.ones, batch_size, self.master.pre.num)

  def update(self, pre_spike):
    x = self.integral(self.x.value, share['t'], share['dt'])
    self.x.value = jnp.where(pre_spike, x - self.U * self.x, x)

  def filter(self, g):
    if jnp.shape(g) != self.x.shape:
      raise ValueError('Shape does not match.')
    return g * self.x

  def __repr__(self):
    return f'{self.__class__.__name__}(tau={self.tau}, U={self.U}, method={self.method})'


class STP(_SynSTP):
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
      U: Union[float, ArrayType] = 0.15,
      tau_f: Union[float, ArrayType] = 1500.,
      tau_d: Union[float, ArrayType] = 200.,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(STP, self).__init__(name=name)
    # parameters
    is_float(tau_f, 'tau_f', min_bound=0, )
    is_float(tau_d, 'tau_d', min_bound=0, )
    is_float(U, 'U', min_bound=0, )
    self.tau_f = tau_f
    self.tau_d = tau_d
    self.U = U
    self.method = method

    # integral function
    self.integral = odeint(self.derivative, method=self.method)

  def clone(self):
    return STP(tau_f=self.tau_f, tau_d=self.tau_d, U=self.U, method=self.method)

  def register_master(self, master):
    super().register_master(master)
    self.x = variable(jnp.ones, self.master.mode, self.master.pre.num)
    self.u = variable(lambda s: jnp.ones(s) * self.U, self.master.mode, self.master.pre.num)

  def reset_state(self, batch_size=None):
    self.x.value = variable(jnp.ones, batch_size, self.master.pre.num)
    self.u.value = variable(lambda s: jnp.ones(s) * self.U, batch_size, self.master.pre.num)

  @property
  def derivative(self):
    du = lambda u, t: self.U - u / self.tau_f
    dx = lambda x, t: (1 - x) / self.tau_d
    return JointEq(du, dx)

  def update(self, pre_spike):
    u, x = self.integral(self.u.value, self.x.value, share['t'], share['dt'])
    u = jnp.where(pre_spike, u + self.U * (1 - self.u), u)
    x = jnp.where(pre_spike, x - u * self.x, x)
    self.x.value = x
    self.u.value = u

  def filter(self, g):
    if jnp.shape(g) != self.x.shape:
      raise ValueError('Shape does not match.')
    return g * self.x * self.u

  def __repr__(self):
    return f'{self.__class__.__name__}(tau_f={self.tau_f}, tau_d={self.tau_d}, U={self.U}, method={self.method})'

