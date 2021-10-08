# -*- coding: utf-8 -*-


import brainpy as bp

bp.math.use_backend('numpy')
bp.integrators.set_default_odeint(method='rk4')


class LIF(bp.NeuGroup):
  def __init__(self, size, t_refractory=1., V_rest=0., V_reset=-5.,
               V_th=20., R=1., tau=10., **kwargs):
    super(LIF, self).__init__(size=size, **kwargs)

    # parameters
    self.V_rest = V_rest
    self.V_reset = V_reset
    self.V_th = V_th
    self.R = R
    self.tau = tau
    self.t_refractory = t_refractory

    # variables
    self.V = bp.math.Variable(bp.math.ones(self.num) * V_rest)
    self.input = bp.math.Variable(bp.math.zeros(self.num))
    self.refractory = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))
    self.spike = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))
    self.t_last_spike = bp.math.Variable(bp.math.ones(self.num) * -1e7)

  @bp.odeint
  def integral(self, V, t, Iext):
    dvdt = (-V + self.V_rest + self.R * Iext) / self.tau
    return dvdt

  def update(self, _t, _dt):
    refractory = (_t - self.t_last_spike) <= self.t_refractory
    V = self.integral(self.V, _t, self.input, dt=_dt)
    V = bp.math.where(refractory, self.V, V)
    spike = self.V_th <= V
    self.t_last_spike[:] = bp.math.where(spike, _t, self.t_last_spike)
    self.V[:] = bp.math.where(spike, self.V_reset, V)
    self.refractory[:] = bp.math.logical_or(refractory, spike)
    self.input[:] = 0.
    self.spike[:] = spike


class HH(bp.NeuGroup):
  def __init__(self, size, ENa=50., gNa=120., EK=-77., gK=36., EL=-54.387,
               gL=0.03, V_th=20., C=1.0, **kwargs):
    super(HH, self).__init__(size=size, **kwargs)

    # parameters
    self.ENa = ENa
    self.EK = EK
    self.EL = EL
    self.gNa = gNa
    self.gK = gK
    self.gL = gL
    self.C = C
    self.V_th = V_th

    # variables
    self.V = bp.math.Variable(-65. * bp.math.ones(self.num))
    self.m = bp.math.Variable(0.5 * bp.math.ones(self.num))
    self.h = bp.math.Variable(0.6 * bp.math.ones(self.num))
    self.n = bp.math.Variable(0.32 * bp.math.ones(self.num))
    self.input = bp.math.Variable(bp.math.zeros(self.num))
    self.spike = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))
    self.t_last_spike = bp.math.Variable(bp.math.ones(self.num) * -1e7)

  @bp.odeint
  def integral(self, V, m, h, n, t, Iext):
    alpha = 0.1 * (V + 40) / (1 - bp.math.exp(-(V + 40) / 10))
    beta = 4.0 * bp.math.exp(-(V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m

    alpha = 0.07 * bp.math.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bp.math.exp(-(V + 35) / 10))
    dhdt = alpha * (1 - h) - beta * h

    alpha = 0.01 * (V + 55) / (1 - bp.math.exp(-(V + 55) / 10))
    beta = 0.125 * bp.math.exp(-(V + 65) / 80)
    dndt = alpha * (1 - n) - beta * n

    I_Na = (self.gNa * m ** 3.0 * h) * (V - self.ENa)
    I_K = (self.gK * n ** 4.0) * (V - self.EK)
    I_leak = self.gL * (V - self.EL)
    dVdt = (- I_Na - I_K - I_leak + Iext) / self.C

    return dVdt, dmdt, dhdt, dndt

  def update(self, _t, _dt):
    V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t, self.input, dt=_dt)
    self.spike[:] = bp.math.logical_and(self.V < self.V_th, V >= self.V_th)
    self.V[:] = V
    self.m[:] = m
    self.h[:] = h
    self.n[:] = n
    self.input[:] = 0.


def test_jit_neugroup_lif():
  bp.math.jit(LIF(10), show_code=True).run(1.)


def test_jit_neugroup_lif_bounded_func():
  bp.integrators.set_default_odeint(method='exponential_euler')
  bp.math.jit(LIF(10).update, show_code=True)


def test_jit_neugroup_lif_integrator():
  bp.integrators.set_default_odeint(method='exponential_euler')
  bp.math.jit(LIF(10).integral, show_code=True)


def test_jit_neugroup_hh():
  bp.math.jit(HH(10), show_code=True).run(1.)


def test_jit_neugroup_hh_bounded_func():
  bp.integrators.set_default_odeint(method='exponential_euler')
  bp.math.jit(HH(10).update, show_code=True)


def test_jit_neugroup_hh_integrator():
  bp.integrators.set_default_odeint(method='exponential_euler')
  bp.math.jit(HH(10).integral, show_code=True)


class LogisticRegression(bp.Base):
  def __init__(self, dimension):
    super(LogisticRegression, self).__init__()

    self.dimension = dimension
    self.w = bp.math.Variable(2.0 * bp.math.ones(dimension) - 1.3)

  def __call__(self, X, Y):
    u = bp.math.dot(((1.0 / (1.0 + bp.math.exp(-Y * bp.math.dot(X, self.w))) - 1.0) * Y), X)
    self.w[:] -= u


def test1():
  num_dim = 10
  num_points = 200
  # num_points = 20000000
  points = bp.math.random.random((num_points, num_dim))
  labels = bp.math.random.random(num_points)
  lr = LogisticRegression(num_dim)
  jit_lr = bp.math.jit(lr, show_code=True)
  print(lr.w)

  num_iter = 20
  for i in range(num_iter):
    jit_lr(points, labels)
  print(lr.w)


