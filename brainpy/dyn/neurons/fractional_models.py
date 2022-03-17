# -*- coding: utf-8 -*-

import brainpy.math as bm
from brainpy.dyn.base import NeuGroup
from brainpy.integrators.fde import CaputoL1Schema
from brainpy.integrators.joint_eq import JointEq
from brainpy.tools.checking import check_float, check_integer

__all__ = [
  'FractionalFHN',
  'FractionalIzhikevich',
]


class FractionalFHN(NeuGroup):
  """


  References
  ----------
  .. [1] Mondal, A., Sharma, S.K., Upadhyay, R.K. *et al.* Firing activities of a fractional-order FitzHugh-Rinzel bursting neuron model and its coupled dynamics. *Sci Rep* **9,** 15721 (2019). https://doi.org/10.1038/s41598-019-52061-4
  """
  def __init__(self, size, alpha, num_step,
               a=0.7, b=0.8, c=-0.775, d=1., delta=0.08, mu=0.0001):
    super(FractionalFHN, self).__init__(size)

    self.alpha = alpha
    self.num_step = num_step

    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.delta = delta
    self.mu = mu

    self.input = bm.Variable(bm.zeros(self.num))
    self.v = bm.Variable(bm.ones(self.num) * 2.5)
    self.w = bm.Variable(bm.zeros(self.num))
    self.y = bm.Variable(bm.zeros(self.num))

    self.integral = CaputoL1Schema(self.derivative,
                                   alpha=alpha,
                                   num_step=num_step,
                                   inits=[self.v, self.w, self.y])

  def dv(self, v, t, w, y):
    return v - v ** 3 / 3 - w + y + self.input

  def dw(self, w, t, v):
    return self.delta * (self.a + v - self.b * w)

  def dy(self, y, t, v):
    return self.mu * (self.c - v - self.d * y)

  @property
  def derivative(self):
    return JointEq([self.dv, self.dw, self.dy])

  def update(self, _t, _dt):
    v, w, y = self.integral(self.v, self.w, self.y, _t, _dt)
    self.v.value = v
    self.w.value = w
    self.y.value = y
    self.input[:] = 0.


class FractionalIzhikevich(NeuGroup):
  """Fractional-order Izhikevich model [10]_.


  References
  ----------
  .. [10] Teka, Wondimu W., Ranjit Kumar Upadhyay, and Argha Mondal. "Spiking and
          bursting patterns of fractional-order Izhikevich model." Communications
          in Nonlinear Science and Numerical Simulation 56 (2018): 161-176.

  """

  def __init__(self, size, num_step, alpha=0.9,
               a=0.02, b=0.20, c=-65., d=8., f=0.04,
               g=5., h=140., tau=1., R=1., V_th=30., name=None):
    # initialization
    super(FractionalIzhikevich, self).__init__(size=size, name=name)

    # params
    self.alpha = alpha
    check_float(alpha, 'alpha', min_bound=0., max_bound=1., allow_none=False, allow_int=True)
    self.a = a
    self.b = b
    self.c = c
    self.d = d
    self.f = f
    self.g = g
    self.h = h
    self.tau = tau
    self.R = R
    self.V_th = V_th

    # variables
    self.V = bm.Variable(bm.ones(self.num) * c)
    self.u = bm.Variable(b * self.V)
    self.input = bm.Variable(bm.zeros(self.num))
    self.spike = bm.Variable(bm.zeros(self.num, dtype=bool))
    self.t_last_spike = bm.Variable(bm.ones(self.num) * -1e7)

    # functions
    check_integer(num_step, 'num_step', allow_none=False)
    self.integral = CaputoL1Schema(f=self.derivative,
                                   alpha=alpha,
                                   num_step=num_step,
                                   inits=[self.V, self.u])

  def dV(self, V, t, u, I_ext):
    dVdt = self.f * V * V + self.g * V + self.h - u + self.R * I_ext
    return dVdt / self.tau

  def du(self, u, t, V):
    dudt = self.a * (self.b * V - u)
    return dudt / self.tau

  @property
  def derivative(self):
    return JointEq([self.dV, self.du])

  def update(self, _t, _dt):
    V, u = self.integral(self.V, self.u, t=_t, I_ext=self.input, dt=_dt)
    spikes = V >= self.V_th
    self.t_last_spike.value = bm.where(spikes, _t, self.t_last_spike)
    self.V.value = bm.where(spikes, self.c, V)
    self.u.value = bm.where(spikes, u + self.d, u)
    self.spike.value = spikes
    self.input[:] = 0.
