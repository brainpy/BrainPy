# -*- coding: utf-8 -*-


import unittest

import brainpy as bp
import matplotlib.pyplot as plt

block = False


class TestGLShortMemory(unittest.TestCase):
  def test_lorenz(self):

    a, b, c = 10, 28, 8 / 3

    def lorenz(x, y, z, t):
      dx = a * (y - x)
      dy = x * (b - z) - y
      dz = x * y - c * z
      return dx, dy, dz

    integral = bp.fde.GLShortMemory(lorenz,
                                    alpha=0.99,
                                    num_memory=500,
                                    inits=[1., 0., 1.])
    runner = bp.integrators.IntegratorRunner(integral,
                                             monitors=list('xyz'),
                                             inits=[1., 0., 1.],
                                             dt=0.005)
    runner.run(100.)

    plt.plot(runner.mon.x.flatten(), runner.mon.z.flatten())
    plt.show(block=block)


