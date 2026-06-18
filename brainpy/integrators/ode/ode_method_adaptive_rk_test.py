# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import unittest

import matplotlib.pyplot as plt
import numpy as np

import brainpy.math as bm
from brainpy.integrators.ode import adaptive_rk

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


def _convergence_order(method_cls, f, exact, y0, t_end=2.0, ns=(20, 40, 80, 160)):
    """Measure the empirical convergence order of the higher-order (B1) solution.

    The integrator is run in *fixed-step* mode (``adaptive=False``) on a
    time-dependent scalar ODE with a known exact solution; the average
    log2 error-ratio over successive step halvings gives the order ``p``.
    """
    errs = []
    for n in ns:
        dt = t_end / n
        intg = method_cls(f, adaptive=False, dt=dt)
        y = y0
        for i in range(n):
            y = intg(y, i * dt, dt=dt)
        errs.append(abs(float(bm.as_jax(y)) - exact(t_end)))
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    return float(np.mean(rates)), errs


class TestRKF45NodeFix(unittest.TestCase):
    """Regression for P6-C1: RKF45 6th-stage node ``c6`` must be 1/2, not 1/3.

    With the wrong node the consistency condition ``sum_j a_{6j} = c_6`` is
    violated and the order conditions break for any time-dependent ``f``,
    collapsing the advertised order-5 solution to order ~1. This is only
    visible when ``f`` depends on ``t`` (autonomous smoke tests miss it).
    """

    def test_rkf45_is_order5_on_time_dependent_ode(self):
        # dy/dt = cos(t), y(0) = 0  ->  y(t) = sin(t).
        # Measure under x64 so the truncation error (not float32 round-off)
        # governs the rate. The genuine order-5 method exceeds order 4; the
        # buggy c6=1/3 variant measures order ~1.
        f = lambda y, t: bm.cos(t)
        exact = lambda t: np.sin(t)
        bm.enable_x64()
        try:
            order, errs = _convergence_order(adaptive_rk.RKF45, f, exact, y0=0.0,
                                             t_end=2.0, ns=(8, 16, 32, 64))
        finally:
            bm.disable_x64()
        self.assertGreater(order, 4.0, msg=f'RKF45 order={order:.2f}, errs={errs}')

    def test_rkf45_node_value(self):
        self.assertAlmostEqual(float(eval(str(adaptive_rk.RKF45.C[-1]))), 0.5)
