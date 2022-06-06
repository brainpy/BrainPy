# -*- coding: utf-8 -*-

from unittest import TestCase

import brainpy as bp

plt = None


class TestIntegratorRunnerForODEs(TestCase):
  def test_ode(self):
    global plt
    if plt is None:
      import matplotlib.pyplot as plt
    sigma = 10
    beta = 8 / 3
    rho = 28

    @bp.odeint(method='rk4', dt=0.001)
    def lorenz(x, y, z, t):
      dx = sigma * (y - x)
      dy = x * (rho - z) - y
      dz = x * y - beta * z
      return dx, dy, dz

    runner = bp.IntegratorRunner(lorenz, monitors=['x', 'y', 'z'], inits=[1., 1., 1.])
    runner.run(100.)
    fig = plt.figure()
    fig.add_subplot(111, projection='3d')
    plt.plot(runner.mon.x[:, 0], runner.mon.y[:, 0], runner.mon.z[:, 0], )
    plt.show()

    runner = bp.IntegratorRunner(lorenz, monitors=['x', 'y', 'z'],
                                 inits=[1., (1., 0.), (1., 0.)])
    runner.run(100.)
    for i in range(2):
      fig = plt.figure()
      fig.add_subplot(111, projection='3d')
      plt.plot(runner.mon.x[:, i], runner.mon.y[:, i], runner.mon.z[:, i])
      plt.show()

  def test_ode2(self):
    a, b, tau = 0.7, 0.8, 12.5
    dV = lambda V, t, w, Iext: V - V * V * V / 3 - w + Iext
    dw = lambda w, t, V: (V + a - b * w) / tau
    fhn = bp.odeint(bp.JointEq([dV, dw]), method='rk4', dt=0.1)

    runner = bp.IntegratorRunner(fhn, monitors=['V', 'w'], inits=[1., 1.], args=dict(Iext=1.5))
    runner.run(100.)
    bp.visualize.line_plot(runner.mon.ts, runner.mon['V'], legend='V')
    bp.visualize.line_plot(runner.mon.ts, runner.mon['w'], legend='w', show=True)

  def test_ode3(self):
    a, b, tau = 0.7, 0.8, 12.5
    dV = lambda V, t, w, Iext: V - V * V * V / 3 - w + Iext
    dw = lambda w, t, V: (V + a - b * w) / tau
    fhn = bp.odeint(bp.JointEq([dV, dw]), method='rk4', dt=0.1)

    Iext, duration = bp.inputs.section_input([0., 1., 0.5], [200, 500, 200], return_length=True)
    runner = bp.IntegratorRunner(fhn,
                                 monitors=['V', 'w'], inits=[1., 1.],
                                 dyn_args=dict(Iext=Iext))
    runner.run(duration)
    bp.visualize.line_plot(runner.mon.ts, runner.mon['V'], legend='V')
    bp.visualize.line_plot(runner.mon.ts, runner.mon['w'], legend='w', show=True)
