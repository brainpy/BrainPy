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

import jax
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

    f_integral = jax.jit(method(f_lorenz, dt=dt))
    x = bm.Variable(bm.ones(1))
    y = bm.Variable(bm.ones(1))
    z = bm.Variable(bm.ones(1))

    def step(t):
        x.value, y.value, z.value = f_integral(x, y, z, t)
        return x.value, y.value, z.value

    times = np.arange(0, duration, dt)

    results = bm.for_loop(
        body_fun=step,
        operands=times,
        jit=True
    )

    mon_x, mon_y, mon_z = results
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
    plt.close()

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
            bm.random.seed()
            mon_x, mon_y, mon_z = run_integrator(method)
            assert np.linalg.norm(mon_x - _baseline_x) / (duration / dt) < 0.1
            assert np.linalg.norm(mon_y - _baseline_y) / (duration / dt) < 0.1
            assert np.linalg.norm(mon_z - _baseline_z) / (duration / dt) < 0.1
