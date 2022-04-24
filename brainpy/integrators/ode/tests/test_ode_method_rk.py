# -*- coding: utf-8 -*-

import unittest

import numpy as np

import brainpy.math as bm
from brainpy.integrators.ode import explicit_rk
plt = None

sigma = 10
beta = 8 / 3
rho = 28
dt = 0.001
duration = 20


def f_lorenz(x, y, z, t):
  dx = sigma * (y - x)
  dy = x * (rho - z) - y
  dz = x * y - beta * z
  return dx, dy, dz


def run_integrator(method, show=False):
  global plt
  if plt is None:
    import matplotlib.pyplot as plt

  f_integral = bm.jit(method(f_lorenz, dt=dt), auto_infer=False)
  x = bm.Variable(bm.ones(1))
  y = bm.Variable(bm.ones(1))
  z = bm.Variable(bm.ones(1))

  def f(t):
    x.value, y.value, z.value = f_integral(x, y, z, t)

  f_scan = bm.make_loop(f, dyn_vars=[x, y, z], out_vars=[x, y, z])

  times = np.arange(0, duration, dt)
  mon_x, mon_y, mon_z = f_scan(times)
  mon_x = np.array(mon_x).flatten()
  mon_y = np.array(mon_y).flatten()
  mon_z = np.array(mon_z).flatten()

  if show:
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.plot(mon_x, mon_y, mon_z)
    ax.set_xlabel('x')
    ax.set_xlabel('y')
    ax.set_xlabel('z')
    plt.show()

  return mon_x, mon_y, mon_z


_baseline_x, _baseline_y, _baseline_z = run_integrator(explicit_rk.RK4)


class TestRKMethods(unittest.TestCase):
  def test_all_methods(self):
    for method in [explicit_rk.Euler,
                   explicit_rk.MidPoint,
                   explicit_rk.Heun2,
                   explicit_rk.Ralston2,
                   explicit_rk.RK2,
                   explicit_rk.RK3,
                   explicit_rk.Heun3,
                   explicit_rk.Ralston3,
                   explicit_rk.SSPRK3,
                   explicit_rk.RK4,
                   explicit_rk.Ralston4,
                   explicit_rk.RK4Rule38]:
      mon_x, mon_y, mon_z = run_integrator(method)
      assert np.linalg.norm(mon_x - _baseline_x) / (duration / dt) < 0.1
      assert np.linalg.norm(mon_y - _baseline_y) / (duration / dt) < 0.1
      assert np.linalg.norm(mon_z - _baseline_z) / (duration / dt) < 0.1
