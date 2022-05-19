# -*- coding: utf-8 -*-


import unittest

import brainpy as bp
import matplotlib.pyplot as plt
from brainpy.integrators.sde.normal import ExponentialEuler

block = True


class TestExpEuler(unittest.TestCase):
  def test1(self):
    p = 0.1

    def lorenz_g(x, y, z, t, **kwargs):
      return p * x, p * y, p * z

    dx = lambda x, t, y, sigma=10: sigma * (y - x)
    dy = lambda y, t, x, z, rho=28: x * (rho - z) - y
    dz = lambda z, t, x, y, beta=8 / 3: x * y - beta * z

    intg = ExponentialEuler(f=bp.JointEq([dx, dy, dz]),
                            g=lorenz_g,
                            intg_type=bp.integrators.ITO_SDE,
                            wiener_type=bp.integrators.SCALAR_WIENER,
                            var_type=bp.integrators.POP_VAR,
                            show_code=True)
    runner = bp.integrators.IntegratorRunner(intg,
                                             monitors=['x', 'y', 'z'],
                                             dt=0.001, inits=[1., 1., 0.])
    runner.run(100.)

    plt.plot(runner.mon.x.flatten(), runner.mon.y.flatten())
    plt.show(block=block)

  def test2(self):
    p = 0.1
    p2 = 0.02

    def lorenz_g(x, y, z, t, **kwargs):
      return bp.math.asarray([p * x, p2 * x]), \
             bp.math.asarray([p * y, p2 * y]), \
             bp.math.asarray([p * z, p2 * z])

    dx = lambda x, t, y, sigma=10: sigma * (y - x)
    dy = lambda y, t, x, z, rho=28: x * (rho - z) - y
    dz = lambda z, t, x, y, beta=8 / 3: x * y - beta * z

    intg = ExponentialEuler(f=bp.JointEq([dx, dy, dz]),
                            g=lorenz_g,
                            intg_type=bp.integrators.ITO_SDE,
                            wiener_type=bp.integrators.VECTOR_WIENER,
                            var_type=bp.integrators.POP_VAR,
                            show_code=True)
    runner = bp.integrators.IntegratorRunner(intg, monitors=['x', 'y', 'z'],
                                             dt=0.001, inits=[1., 1., 0.], jit=False)
    with self.assertRaises(ValueError):
      runner.run(100.)

    plt.plot(runner.mon.x.flatten(), runner.mon.y.flatten())
    plt.show(block=block)
    # bp.visualize.line_plot(runner.mon.x, runner.mon.y, show=True)

  def test3(self):
    p = 0.1
    p2 = 0.02

    def lorenz_g(x, y, z, t, **kwargs):
      return bp.math.asarray([p * x, p2 * x]).T, \
             bp.math.asarray([p * y, p2 * y]).T, \
             bp.math.asarray([p * z, p2 * z]).T

    dx = lambda x, t, y, sigma=10: sigma * (y - x)
    dy = lambda y, t, x, z, rho=28: x * (rho - z) - y
    dz = lambda z, t, x, y, beta=8 / 3: x * y - beta * z

    intg = ExponentialEuler(f=bp.JointEq([dx, dy, dz]),
                            g=lorenz_g,
                            intg_type=bp.integrators.ITO_SDE,
                            wiener_type=bp.integrators.VECTOR_WIENER,
                            var_type=bp.integrators.POP_VAR,
                            show_code=True)
    runner = bp.integrators.IntegratorRunner(intg,
                                             monitors=['x', 'y', 'z'],
                                             dt=0.001,
                                             inits=[1., 1., 0.],
                                             jit=True)
    runner.run(100.)

    plt.plot(runner.mon.x.flatten(), runner.mon.y.flatten())
    plt.show(block=block)


class TestMilstein(unittest.TestCase):
  def test1(self):
    p = 0.1
    sigma = 10
    rho = 28
    beta = 8 / 3

    gx = lambda x, t, y: p * x
    gy = lambda y, t, x, z: p * y
    gz = lambda z, t, x, y: p * z

    fx = lambda x, t, y: sigma * (y - x)
    fy = lambda y, t, x, z: x * (rho - z) - y
    fz = lambda z, t, x, y: x * y - beta * z

    intg = bp.sdeint(f=bp.JointEq(fx, fy, fz),
                     g=bp.JointEq(gx, gy, gz),
                     intg_type=bp.integrators.ITO_SDE,
                     wiener_type=bp.integrators.SCALAR_WIENER,
                     var_type=bp.integrators.POP_VAR,
                     method='milstein')
    runner = bp.integrators.IntegratorRunner(intg,
                                             monitors=['x', 'y', 'z'],
                                             dt=0.001, inits=[1., 1., 0.],
                                             jit=True)
    runner.run(100.)

    plt.plot(runner.mon.x.flatten(), runner.mon.y.flatten())
    plt.show(block=block)
