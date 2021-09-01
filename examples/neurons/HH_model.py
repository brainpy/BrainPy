# -*- coding: utf-8 -*-

import sys

sys.path.append(r'/mnt/d/codes/Projects/BrainPy')

import brainpy as bp

bp.math.use_backend('numpy')
bp.math.set_dt(dt=0.02)


class HH(bp.NeuGroup):
  def __init__(self, size, ENa=50., EK=-77., EL=-54.387,
               C=1.0, gNa=120., gK=36., gL=0.03, V_th=20.,
               **kwargs):
    super(HH, self).__init__(size=size, **kwargs)

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
    self.V = bp.math.Variable(bp.math.ones(self.num) * -65.)
    self.m = bp.math.Variable(bp.math.ones(self.num) * 0.5)
    self.h = bp.math.Variable(bp.math.ones(self.num) * 0.6)
    self.n = bp.math.Variable(bp.math.ones(self.num) * 0.32)
    self.spike = bp.math.Variable(bp.math.zeros(self.num, dtype=bool))
    self.input = bp.math.Variable(bp.math.zeros(self.num))

  # @bp.odeint(method='exponential_euler')
  @bp.odeint(method='rk4')
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


def run_hh1():
  group = bp.math.jit(HH(int(1e4), monitors=['V']))

  group.run(200., inputs=('input', 10.), report=0.1)
  bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)

  group.run(200., report=0.1)
  bp.visualize.line_plot(group.mon.ts, group.mon.V, show=True)


def run_hh2():
  group = bp.math.jit(HH(100, monitors=['V']))

  group.run(200., inputs=('input', 10.), report=0.2)
  bp.visualize.line_plot(group.mon.ts, group.mon['V'], show=True)


if __name__ == '__main__':
  # run_hh2()
  run_hh1()
  # run_hh_with_interval_monitor()


def run_hh_with_interval_monitor():
  group = HH(100,
             monitors=bp.simulation.Monitor(variables=['V'],
                                            every=[1.]))

  group.run(200. * 50, inputs=('input', 10.), report=True)
  bp.visualize.line_plot(group.mon['V.t'], group.mon.V, show=True)


class HH_with_Every(bp.NeuGroup):
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
    self.spike = bp.math.zeros(size)
    self.input = 0.

    super(HH_with_Every, self).__init__(size=size, **kwargs)

  @staticmethod
  @bp.odeint(method='exponential_euler')
  def integral(V, m, h, n, t, Iext, gNa, ENa, gK, EK, gL, EL, C):
    alpha = 0.1 * (V + 40) / (1 - bp.math.exp(-(V + 40) / 10))
    beta = 4.0 * bp.math.exp(-(V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m

    alpha = 0.07 * bp.math.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bp.math.exp(-(V + 35) / 10))
    dhdt = alpha * (1 - h) - beta * h

    alpha = 0.01 * (V + 55) / (1 - bp.math.exp(-(V + 55) / 10))
    beta = 0.125 * bp.math.exp(-(V + 65) / 80)
    dndt = alpha * (1 - n) - beta * n

    I_Na = (gNa * m ** 3.0 * h) * (V - ENa)
    I_K = (gK * n ** 4.0) * (V - EK)
    I_leak = gL * (V - EL)
    dVdt = (- I_Na - I_K - I_leak + Iext) / C

    return dVdt, dmdt, dhdt, dndt

  @bp.every(time=1.)
  def update(self, _t, _i, _dt):
    V, m, h, n = self.integral(self.V, self.m, self.h, self.n, _t,
                               self.input, self.gNa, self.ENa, self.gK,
                               self.EK, self.gL, self.EL, self.C)
    self.spike = (self.V < self.V_th) * (V >= self.V_th)
    self.V = V
    self.m = m
    self.h = h
    self.n = n


def run_hh2_with_interval_monitor():
  # group = HH(100, monitors=bp.simulation.Monitor(variables=['V'], every=[1.]))
  group = HH_with_Every(100, monitors=bp.simulation.Monitor(variables=['V'], every=[1.]))
  group.input = 10.

  group.run(200. * (1 / bp.math.get_dt()), report=True)
  bp.visualize.line_plot(group.mon['V.t'], group.mon.V, show=True)

  group.input = 0.
  group.run(200. * (1 / bp.math.get_dt()), report=True)
  bp.visualize.line_plot(group.mon['V.t'], group.mon.V, show=True)


if __name__ == '__main__1':
  run_hh2_with_interval_monitor()
