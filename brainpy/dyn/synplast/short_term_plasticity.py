# -*- coding: utf-8 -*-

from typing import Union

import brainpy.math as bm
from brainpy.dyn.base import SynSTP
from brainpy.integrators import odeint, JointEq
from brainpy.tools.checking import check_float
from brainpy.types import Array
from brainpy.initialize import variable

__all__ = [
  'STD',
  'STP',
]


class STD(SynSTP):
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
    super(STD, self).__init__(name=name)

    # parameters
    check_float(tau, 'tau', min_bound=0, )
    check_float(U, 'U', min_bound=0, )
    self.tau = tau
    self.U = U
    self.method = method

    # integral function
    self.integral = odeint(lambda x, t: (1 - x) / self.tau, method=self.method)

  def register_master(self, master):
    super(STD, self).register_master(master)

    # variables
    self.x = variable(bm.ones, self.master.mode, self.master.pre.num)

  def reset_state(self, batch_size=None):
    self.x.value = variable(bm.ones, batch_size, self.master.pre.num)

  def update(self, tdi, pre_spike):
    x = self.integral(self.x.value, tdi['t'], tdi['dt'])
    self.x.value = bm.where(pre_spike, x - self.U * self.x, x)

  def filter(self, g):
    if bm.shape(g) != self.x.shape:
      raise ValueError('Shape does not match.')
    return g * self.x


class STP(SynSTP):
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
      U: Union[float, Array] = 0.15,
      tau_f: Union[float, Array] = 1500.,
      tau_d: Union[float, Array] = 200.,
      method: str = 'exp_auto',
      name: str = None
  ):
    super(STP, self).__init__(name=name)
    # parameters
    check_float(tau_f, 'tau_f', min_bound=0, )
    check_float(tau_d, 'tau_d', min_bound=0, )
    check_float(U, 'U', min_bound=0, )
    self.tau_f = tau_f
    self.tau_d = tau_d
    self.U = U
    self.method = method

    # integral function
    self.integral = odeint(self.derivative, method=self.method)

  def register_master(self, master):
    super(STP, self).register_master(master)

    # variables
    self.x = variable(bm.ones, self.master.mode, self.master.pre.num)
    self.u = variable(lambda s: bm.ones(s) * self.U, self.master.mode, self.master.pre.num)

  def reset_state(self, batch_size=None):
    self.x.value = variable(bm.ones, batch_size, self.master.pre.num)
    self.u.value = variable(lambda s: bm.ones(s) * self.U, batch_size, self.master.pre.num)

  @property
  def derivative(self):
    du = lambda u, t: self.U - u / self.tau_f
    dx = lambda x, t: (1 - x) / self.tau_d
    return JointEq([du, dx])

  def update(self, tdi, pre_spike):
    u, x = self.integral(self.u.value, self.x.value, tdi['t'], tdi['dt'])
    u = bm.where(pre_spike, u + self.U * (1 - self.u), u)
    x = bm.where(pre_spike, x - u * self.x, x)
    self.x.value = x
    self.u.value = u

  def filter(self, g):
    if bm.shape(g) != self.x.shape:
      raise ValueError('Shape does not match.')
    return g * self.x * self.u
