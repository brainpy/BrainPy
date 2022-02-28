# -*- coding: utf-8 -*-


from unittest import TestCase

import matplotlib.pyplot as plt

import brainpy as bp
import brainpy.math as bm


class ExampleDS(bp.dyn.DynamicalSystem):
  def __init__(self):
    super(ExampleDS, self).__init__()
    self.i = bm.Variable(bm.zeros(1))
    self.o = bm.Variable(bm.zeros(2))

  def update(self, _t, _dt):
    self.i += 1


# class TestMonitor(TestCase):
#   def test_1d_array(self):
#     try1 = TryGroup(monitors=['a'])
#     try1.a = np.ones(1)
#     try1.run(100.)
#
#     assert np.ndim(try1.mon.a) == 2 and np.shape(try1.mon.a)[1] == 1
#     assert np.allclose(np.arange(2, 1002).reshape((-1, 1)), try1.mon.a)
#
#   def test_2d_array():
#     set(dt=0.1)
#     try1 = TryGroup(monitors=['a'])
#     try1.a = np.ones((2, 2))
#     try1.run(100.)
#
#     assert np.ndim(try1.mon.a) == 2 and np.shape(try1.mon.a)[1] == 4
#     series = np.arange(2, 1002).reshape((-1, 1))
#     series = np.repeat(series, 4, axis=1)
#     assert np.allclose(series, try1.mon.a)
#
#   def test_monitor_with_every():
#     set(dt=0.1)
#
#     # try1: 2d array
#     try1 = TryGroup(monitors=Monitor(variables=['a'], every=[1.]))
#     try1.run(100.)
#     assert np.ndim(try1.mon.a) == 2 and np.shape(try1.mon.a)[1] == 4
#     series = np.arange(2, 1002, 1. / 0.1).reshape((-1, 1))
#     series = np.repeat(series, 4, axis=1)
#     assert np.allclose(series, try1.mon.a)
#
#     # try2: 1d array
#     try2 = TryGroup(monitors=Monitor(variables=['a'], every=[1.]))
#     try2.a = np.array([1., 1.])
#     try2.run(100.)
#     assert np.ndim(try2.mon.a) == 2 and np.shape(try2.mon.a)[1] == 2
#     series = np.arange(2, 1002, 1. / 0.1).reshape((-1, 1))
#     series = np.repeat(series, 2, axis=1)
#     assert np.allclose(series, try2.mon.a)
#
#     # try2: scalar
#     try3 = TryGroup(monitors=Monitor(variables=['a'], every=[1.]))
#     try3.a = 1.
#     try3.run(100.)
#     assert np.ndim(try3.mon.a) == 2 and np.shape(try3.mon.a)[1] == 1
#     series = np.arange(2, 1002, 1. / 0.1).reshape((-1, 1))
#     assert np.allclose(series, try3.mon.a)


class TestIntegratorRunner(TestCase):
  def test_ode(self):
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
    runner = bp.IntegratorRunner(fhn, monitors=['V', 'w'], inits=[1., 1.],
                                 dyn_args=dict(Iext=Iext))
    runner.run(duration)
    bp.visualize.line_plot(runner.mon.ts, runner.mon['V'], legend='V')
    bp.visualize.line_plot(runner.mon.ts, runner.mon['w'], legend='w', show=True)
