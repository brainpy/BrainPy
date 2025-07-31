# -*- coding: utf-8 -*-

import unittest

import numpy as np
import matplotlib.pyplot as plt

import brainpy.math as bm
from brainpy._src.integrators.ode import adaptive_rk

sigma = 10
beta = 8 / 3
rho = 28
_dt = 0.001
duration = 20


def f_lorenz(x, y, z, t):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz


def run_integrator(method, show=False, tol=0.001, adaptive=True):
    f_integral = method(f_lorenz, adaptive=adaptive, tol=tol, show_code=True)
    x = bm.Variable(bm.ones(1))
    y = bm.Variable(bm.ones(1))
    z = bm.Variable(bm.ones(1))
    dt = bm.Variable(bm.ones(1) * 0.01)

    def step(t):
        x_new, y_new, z_new, dt_new = f_integral(x, y, z, t, dt=dt.value)
        x.value = x_new
        y.value = y_new
        z.value = z_new
        dt.value = dt_new

        return x.value, y.value, z.value, dt.value

    times = bm.arange(0, duration, _dt)
    results = bm.for_loop(
        body_fun=step,
        operands=times.value,
        jit=True
    )

    mon_x, mon_y, mon_z, mon_dt = results
    mon_x = np.array(mon_x).flatten()
    mon_y = np.array(mon_y).flatten()
    mon_z = np.array(mon_z).flatten()
    mon_dt = np.array(mon_dt).flatten()

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.plot(mon_x, mon_y, mon_z)
        ax.set_xlabel('x')
        ax.set_xlabel('y')
        ax.set_xlabel('z')
        plt.show()

        plt.plot(mon_dt)
        plt.show()

    return mon_x, mon_y, mon_z, mon_dt


class TestAdaptiveRK(unittest.TestCase):
    def test_all_methods(self):
        for method in [adaptive_rk.RKF12,
                       adaptive_rk.RKF45,
                       adaptive_rk.DormandPrince,
                       adaptive_rk.CashKarp,
                       adaptive_rk.BogackiShampine,
                       adaptive_rk.HeunEuler]:
            bm.random.seed()
            run_integrator(method, show=False)
