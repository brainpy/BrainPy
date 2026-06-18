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
from unittest import TestCase

import matplotlib.pyplot as plt

import brainpy as bp

show = False


class TestIntegratorRunnerForODEs(TestCase):
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
        if show: plt.show()

        runner = bp.IntegratorRunner(lorenz,
                                     monitors=['x', 'y', 'z'],
                                     inits=[1., (1., 0.), (1., 0.)])
        runner.run(100.)
        for i in range(2):
            fig = plt.figure()
            fig.add_subplot(111, projection='3d')
            plt.plot(runner.mon.x[:, i], runner.mon.y[:, i], runner.mon.z[:, i])
            plt.show()

        plt.close()

    def test_ode2(self):
        a, b, tau = 0.7, 0.8, 12.5
        dV = lambda V, t, w, Iext: V - V * V * V / 3 - w + Iext
        dw = lambda w, t, V: (V + a - b * w) / tau
        fhn = bp.odeint(bp.JointEq([dV, dw]), method='rk4', dt=0.1)

        runner = bp.IntegratorRunner(fhn, monitors=['V', 'w'], inits=[1., 1.])
        runner.run(100., args=dict(Iext=1.5))
        bp.visualize.line_plot(runner.mon.ts, runner.mon['V'], legend='V')
        bp.visualize.line_plot(runner.mon.ts, runner.mon['w'], legend='w', show=show)
        plt.close()

    def test_ode3(self):
        a, b, tau = 0.7, 0.8, 12.5
        dV = lambda V, t, w, Iext: V - V * V * V / 3 - w + Iext
        dw = lambda w, t, V: (V + a - b * w) / tau
        fhn = bp.odeint(bp.JointEq([dV, dw]), method='rk4', dt=0.1)

        Iext, duration = bp.inputs.section_input([0., 1., 0.5], [200, 500, 200], return_length=True)
        runner = bp.IntegratorRunner(fhn,
                                     monitors=['V', 'w'],
                                     inits=[1., 1.])
        runner.run(duration, dyn_args=dict(Iext=Iext))
        bp.visualize.line_plot(runner.mon.ts, runner.mon['V'], legend='V')
        bp.visualize.line_plot(runner.mon.ts, runner.mon['w'], legend='w', show=show)
        plt.close()

    def test_ode_continuous_run(self):
        a, b, tau = 0.7, 0.8, 12.5
        dV = lambda V, t, w, Iext: V - V * V * V / 3 - w + Iext
        dw = lambda w, t, V: (V + a - b * w) / tau
        fhn = bp.odeint(bp.JointEq([dV, dw]), method='rk4', dt=0.1)

        runner = bp.IntegratorRunner(fhn,
                                     monitors=['V', 'w'],
                                     inits=[1., 1.])
        Iext, duration = bp.inputs.section_input([0., 1., 0.5], [200, 200, 200], return_length=True)
        runner.run(duration, dyn_args=dict(Iext=Iext))
        bp.visualize.line_plot(runner.mon.ts, runner.mon['V'], legend='V')
        bp.visualize.line_plot(runner.mon.ts, runner.mon['w'], legend='w')

        Iext, duration = bp.inputs.section_input([0.5], [200], return_length=True)
        runner.run(duration, dyn_args=dict(Iext=Iext))
        bp.visualize.line_plot(runner.mon.ts, runner.mon['V'], legend='V-run2')
        bp.visualize.line_plot(runner.mon.ts, runner.mon['w'], legend='w-run2', show=show)
        plt.close()

    def test_ode_dyn_args(self):
        a, b, tau = 0.7, 0.8, 12.5
        dV = lambda V, t, w, Iext: V - V * V * V / 3 - w + Iext
        dw = lambda w, t, V: (V + a - b * w) / tau
        fhn = bp.odeint(bp.JointEq([dV, dw]), method='rk4', dt=0.1)

        Iext, duration = bp.inputs.section_input([0., 1., 0.5],
                                                 [200, 500, 199],
                                                 return_length=True)
        runner = bp.IntegratorRunner(fhn,
                                     monitors=['V', 'w'],
                                     inits=[1., 1.])
        with self.assertRaises(ValueError):
            runner.run(duration + 1, dyn_args=dict(Iext=Iext))

        plt.close()
