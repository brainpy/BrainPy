# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

import brainpy as bp
import brainpy.math as bm


class HH(bp.dyn.NeuGroup):
  def __init__(self, size, ENa=50., gNa=120., EK=-77., gK=36., EL=-54.387, gL=0.03,
               V_th=20., C=1.0, name=None):
    super(HH, self).__init__(size=size, name=name)

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
    self.V = bm.Variable(bm.ones(self.num) * -65.)
    self.m = bm.Variable(0.5 * bm.ones(self.num))
    self.h = bm.Variable(0.6 * bm.ones(self.num))
    self.n = bm.Variable(0.32 * bm.ones(self.num))
    self.spike = bm.Variable(bm.zeros(size, dtype=bool))
    self.input = bm.Variable(bm.zeros(size))

    # integral functions
    self.int_h = bp.ode.ExponentialEuler(self.dh)
    self.int_n = bp.ode.ExponentialEuler(self.dn)
    self.int_m = bp.ode.ExponentialEuler(self.dm)
    self.int_V = bp.ode.ExponentialEuler(self.dV)

  def dh(self, h, t, V):
    alpha = 0.07 * bm.exp(-(V + 65) / 20.)
    beta = 1 / (1 + bm.exp(-(V + 35) / 10))
    dhdt = alpha * (1 - h) - beta * h
    return dhdt

  def dn(self, n, t, V):
    alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
    beta = 0.125 * bm.exp(-(V + 65) / 80)
    dndt = alpha * (1 - n) - beta * n
    return dndt

  def dm(self, m, t, V):
    alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
    beta = 4.0 * bm.exp(-(V + 65) / 18)
    dmdt = alpha * (1 - m) - beta * m
    return dmdt

  def dV(self, V, t, m, h, n, Iext):
    I_Na = (self.gNa * m ** 3.0 * h) * (V - self.ENa)
    I_K = (self.gK * n ** 4.0) * (V - self.EK)
    I_leak = self.gL * (V - self.EL)
    dVdt = (- I_Na - I_K - I_leak + Iext) / self.C
    return dVdt

  def step(self, h, Iext):
    V, m, h, n = bm.split(h, 4)
    dV = self.dV(V, 0., m, h, n, Iext)
    dm = self.dm(m, 0., V)
    dh = self.dh(h, 0., V)
    dn = self.dn(n, 0., V)
    return bm.concatenate([dV, dm, dh, dn])

  def update(self, t, dt):
    m = self.int_m(self.m, t, self.V, dt=dt)
    h = self.int_h(self.h, t, self.V, dt=dt)
    n = self.int_n(self.n, t, self.V, dt=dt)
    V = self.int_V(self.V, t, self.m, self.h, self.n, self.input, dt=dt)
    self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
    self.V.value = V
    self.h.value = h
    self.n.value = n
    self.m.value = m
    self.input[:] = 0.


model = HH(1)
I = 5.
run = bp.dyn.DSRunner(model, inputs=('input', I), monitors=['V'])
run(100)
bp.visualize.line_plot(run.mon.ts, run.mon.V, legend='V', show=True)

# analysis
finder = bp.analysis.SlowPointFinder(lambda h: model.step(h, I))
V = bm.random.normal(0., 5., (1000, model.num)) - 50.
mhn = bm.random.random((1000, model.num * 3))
finder.find_fps_with_opt_solver(candidates=bm.hstack([V, mhn]))
finder.filter_loss(1e-7)
finder.keep_unique()
print('fixed_points: ', finder.fixed_points)
print('losses:', finder.losses)
if len(finder.fixed_points):
  jac = finder.compute_jacobians(finder.fixed_points)
  for i in range(len(finder.fixed_points)):
    eigval, eigvec = np.linalg.eig(np.asarray(jac[i]))
    plt.figure()
    plt.scatter(np.real(eigval), np.imag(eigval))
    plt.plot([0, 0], [-1, 1], '--')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.title(f'FP {i}')
    plt.show()

# verify
for i, fp in enumerate(finder.fixed_points):
  model.V[:] = fp[0]
  model.m[:] = fp[1]
  model.h[:] = fp[2]
  model.n[:] = fp[3]
  run = bp.dyn.DSRunner(model, inputs=('input', I), monitors=['V'])
  run(100)
  bp.visualize.line_plot(run.mon.ts, run.mon.V, legend='V', title=f'FP {i}', show=True)
