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

import brainpy as bp
import brainpy.math as bm
from brainpy.integrators.sde.normal import ExponentialEuler

show = False


class TestStratonovichEulerHeun(unittest.TestCase):
    def test_geometric_bm_weak_convergence(self):
        # Regression for C2 (audit 2026-07-08): the Stratonovich Euler-Heun predictor
        # must use the *scaled* Wiener increment (noise * sqrt(dt)), not the raw
        # standard-normal noise. With the bug the predictor half-step is off by a
        # factor sqrt(dt), which inflates the effective drift by ~0.5*b^2/sqrt(dt) per
        # step and blows the ensemble mean far above the analytical value.
        #
        # For the Stratonovich GBM  dX = a*X dt + b*X o dW  the Ito-equivalent drift is
        # (a + b^2/2), so  E[X(T)] = X0 * exp((a + b^2/2) * T).
        bm.random.seed(20260708)
        a, b, X0, T, dt, M = 0.0, 0.5, 1.0, 1.0, 0.01, 4000
        intg = bp.sdeint(f=lambda x, t: a * x, g=lambda x, t: b * x, method='heun',
                         intg_type=bp.integrators.STRA_SDE,
                         wiener_type=bp.integrators.SCALAR_WIENER)
        runner = bp.IntegratorRunner(intg, monitors=['x'], dt=dt,
                                     inits={'x': X0 * np.ones(M)}, progress_bar=False)
        runner.run(T)
        mean = float(np.asarray(bm.as_jax(runner.mon.x))[-1].mean())
        analytical = X0 * np.exp((a + 0.5 * b * b) * T)  # ~1.133
        buggy = X0 * np.exp(a * T + 0.5 * b * b * np.sqrt(dt) * (T / dt))  # ~3.49
        # Within Monte-Carlo error of the analytical value ...
        self.assertLess(abs(mean - analytical), 0.15,
                        msg=f'mean={mean} analytical={analytical}')
        # ... and nowhere near the (much larger) value the buggy predictor produces.
        self.assertLess(mean, 0.5 * (analytical + buggy))


class TestExpEuler(unittest.TestCase):
    def test1(self):
        p = 0.1

        def lorenz_g(x, y, z, t, **kwargs):
            return p * x, p * y, p * z

        dx = lambda x, t, y, sigma=10: sigma * (y - x)
        dy = lambda y, t, x, z, rho=28: x * (rho - z) - y
        dz = lambda z, t, x, y, beta=8 / 3: x * y - beta * z

        bm.random.seed()
        intg = ExponentialEuler(f=bp.JointEq([dx, dy, dz]),
                                g=lorenz_g,
                                intg_type=bp.integrators.ITO_SDE,
                                wiener_type=bp.integrators.SCALAR_WIENER,
                                var_type=bp.integrators.POP_VAR,
                                show_code=True)
        runner = bp.IntegratorRunner(intg,
                                     monitors=['x', 'y', 'z'],
                                     dt=0.001, inits=[1., 1., 0.])
        runner.run(100.)

        plt.plot(runner.mon.x.flatten(), runner.mon.y.flatten())
        if show:
            plt.show()
        plt.close()

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

        bm.random.seed()
        intg = ExponentialEuler(f=bp.JointEq([dx, dy, dz]),
                                g=lorenz_g,
                                intg_type=bp.integrators.ITO_SDE,
                                wiener_type=bp.integrators.VECTOR_WIENER,
                                var_type=bp.integrators.POP_VAR,
                                show_code=True)
        runner = bp.IntegratorRunner(intg, monitors=['x', 'y', 'z'],
                                     dt=0.001, inits=[1., 1., 0.], jit=False)
        with self.assertRaises(ValueError):
            runner.run(100.)

    def test3(self):
        p = 0.1
        p2 = 0.02

        def lorenz_g(x, y, z, t, **kwargs):
            return bp.math.asarray([p * x, p2 * x]).T, \
                bp.math.asarray([p * y, p2 * y]).T, \
                bp.math.asarray([p * z, p2 * z]).T

        bm.random.seed()
        dx = lambda x, t, y, sigma=10: sigma * (y - x)
        dy = lambda y, t, x, z, rho=28: x * (rho - z) - y
        dz = lambda z, t, x, y, beta=8 / 3: x * y - beta * z

        intg = ExponentialEuler(f=bp.JointEq([dx, dy, dz]),
                                g=lorenz_g,
                                intg_type=bp.integrators.ITO_SDE,
                                wiener_type=bp.integrators.VECTOR_WIENER,
                                var_type=bp.integrators.POP_VAR,
                                show_code=True)
        runner = bp.IntegratorRunner(intg,
                                     monitors=['x', 'y', 'z'],
                                     dt=0.001,
                                     inits=[1., 1., 0.],
                                     jit=True)
        runner.run(100.)

        plt.plot(runner.mon.x.flatten(), runner.mon.y.flatten())
        if show:
            plt.show()
        plt.close()


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

        bm.random.seed()
        intg = bp.sdeint(f=bp.JointEq(fx, fy, fz),
                         g=bp.JointEq(gx, gy, gz),
                         intg_type=bp.integrators.ITO_SDE,
                         wiener_type=bp.integrators.SCALAR_WIENER,
                         var_type=bp.integrators.POP_VAR,
                         method='milstein')
        runner = bp.IntegratorRunner(intg,
                                     monitors=['x', 'y', 'z'],
                                     dt=0.001, inits=[1., 1., 0.],
                                     jit=True)
        runner.run(100.)

        plt.plot(runner.mon.x.flatten(), runner.mon.y.flatten())
        if show:
            plt.show()
        plt.close()
