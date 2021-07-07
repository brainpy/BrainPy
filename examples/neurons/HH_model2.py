# -*- coding: utf-8 -*-

import sys

sys.path.append(r'/mnt/d/codes/Projects/BrainPy')

import brainpy as bp

bp.math.use_backend('jax')
bp.math.set_dt(dt=0.1)


class HH(bp.NeuGroup):
  target_backend = 'general'

  def __init__(self, size, ENa=50., EK=-77., EL=-54.387,
               C=1.0, gNa=120., gK=36., gL=0.03, V_th=20.,
               **kwargs):
    # parameters
    self.ENa = ENa
    self.EK = EK
    self.EL = EL
    self.C = C
    self.gNa = gNa
    self.gK = gK
    self.gL = gL
    self.V_th = V_th

    # variables
    self.V = bp.math.ones(size) * -65.
    self.m = bp.math.ones(size) * 0.5
    self.h = bp.math.ones(size) * 0.6
    self.n = bp.math.ones(size) * 0.32
    self.spike = bp.math.zeros(size, dtype=bool)
    self.input = bp.math.zeros(size)

    super(HH, self).__init__(size=size, **kwargs)

  @bp.odeint(method='exponential_euler')
  # @bp.odeint(method='rk4')
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

  def update(self, _t, _i):
    V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t, self.input)
    self.spike[:] = (self.V < self.V_th) * (V >= self.V_th)
    self.V[:] = V
    self.m[:] = m
    self.h[:] = h
    self.n[:] = n
    self.input[:] = 0.


class HH2(bp.NeuGroup):
  target_backend = 'general'

  def __init__(self, size, ENa=50., EK=-77., EL=-54.387,
               C=1.0, gNa=120., gK=36., gL=0.03, V_th=20.,
               **kwargs):
    # parameters
    self.ENa = ENa
    self.EK = EK
    self.EL = EL
    self.C = C
    self.gNa = gNa
    self.gK = gK
    self.gL = gL
    self.V_th = V_th

    # variables
    self.V = bp.math.ones(size) * -65.
    self.m = bp.math.ones(size) * 0.5
    self.h = bp.math.ones(size) * 0.6
    self.n = bp.math.ones(size) * 0.32
    self.spike = bp.math.zeros(size, dtype=bool)
    self.input = bp.math.zeros(size)

    super(HH2, self).__init__(size=size, **kwargs)

  @bp.odeint(method='exponential_euler')
  # @bp.odeint(method='rk4')
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

  def update(self, _t, _i):
    V, m, h, n = self.integral(self.V.value, self.m.value,
                               self.h.value, self.n.value, _t,
                               self.input.value)
    self.spike.value = (self.V.value < self.V_th) * (V >= self.V_th)
    self.V.value = V
    self.m.value = m
    self.h.value = h
    self.n.value = n
    self.input.value = bp.math.zeros(self.num)


def run_hh1():
  size = int(1e5)

  group = bp.math.jit(HH(size, monitors=['V']))
  group.run(200., inputs=('input', 10.), report=0.5)
  bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

  group = bp.math.jit(HH2(size, monitors=['V']))
  group.run(200., inputs=('input', 10.), report=0.5)
  bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)


if __name__ == '__main__':
  run_hh1()
