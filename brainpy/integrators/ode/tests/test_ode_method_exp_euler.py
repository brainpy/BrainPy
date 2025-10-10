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

import brainpy as bp
import brainpy.math as bm
from brainpy.integrators.ode.exponential import ExponentialEuler

block = False


class TestExpnentialEuler(unittest.TestCase):
    def test_hh_model(self):
        def drivative(V, m, h, n, t, Iext, gNa, ENa, gK, EK, gL, EL, C):
            alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
            beta = 4.0 * bm.exp(-(V + 65) / 18)
            dmdt = alpha * (1 - m) - beta * m

            alpha = 0.07 * bm.exp(-(V + 65) / 20.)
            beta = 1 / (1 + bm.exp(-(V + 35) / 10))
            dhdt = alpha * (1 - h) - beta * h

            alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
            beta = 0.125 * bm.exp(-(V + 65) / 80)
            dndt = alpha * (1 - n) - beta * n

            I_Na = (gNa * m ** 3.0 * h) * (V - ENa)
            I_K = (gK * n ** 4.0) * (V - EK)
            I_leak = gL * (V - EL)
            dVdt = (- I_Na - I_K - I_leak + Iext) / C

            return dVdt, dmdt, dhdt, dndt

        with self.assertRaises(bp.errors.DiffEqError):
            ExponentialEuler(f=drivative, show_code=True, dt=0.01, var_type='SCALAR')

    def test1(self):
        def dev(x, t):
            dx = bm.power(x, 3)
            return dx

        ExponentialEuler(f=dev, show_code=True, dt=0.01)


class TestExpEulerAuto(unittest.TestCase):
    def test_hh_model(self):
        class HH(bp.dyn.NeuDyn):
            def __init__(self, size, ENa=55., EK=-90., EL=-65, C=1.0, gNa=35., gK=9.,
                         gL=0.1, V_th=20., phi=5.0, name=None, method='exponential_euler'):
                super(HH, self).__init__(size=size, name=name)

                # parameters
                self.ENa = ENa
                self.EK = EK
                self.EL = EL
                self.C = C
                self.gNa = gNa
                self.gK = gK
                self.gL = gL
                self.V_th = V_th
                self.phi = phi

                # variables
                self.V = bm.Variable(bm.ones(size) * -65.)
                self.h = bm.Variable(bm.ones(size) * 0.6)
                self.n = bm.Variable(bm.ones(size) * 0.32)
                self.spike = bm.Variable(bm.zeros(size, dtype=bool))
                self.input = bm.Variable(bm.zeros(size))

                self.integral = bp.odeint(bp.JointEq(self.dV, self.dh, self.dn), method=method, show_code=True)

            def dh(self, h, t, V):
                alpha = 0.07 * bm.exp(-(V + 58) / 20)
                beta = 1 / (bm.exp(-0.1 * (V + 28)) + 1)
                dhdt = self.phi * (alpha * (1 - h) - beta * h)
                return dhdt

            def dn(self, n, t, V):
                alpha = -0.01 * (V + 34) / (bm.exp(-0.1 * (V + 34)) - 1)
                beta = 0.125 * bm.exp(-(V + 44) / 80)
                dndt = self.phi * (alpha * (1 - n) - beta * n)
                return dndt

            def dV(self, V, t, h, n, Iext):
                m_alpha = -0.1 * (V + 35) / (bm.exp(-0.1 * (V + 35)) - 1)
                m_beta = 4 * bm.exp(-(V + 60) / 18)
                m = m_alpha / (m_alpha + m_beta)
                INa = self.gNa * m ** 3 * h * (V - self.ENa)
                IK = self.gK * n ** 4 * (V - self.EK)
                IL = self.gL * (V - self.EL)
                dVdt = (- INa - IK - IL + Iext) / self.C

                return dVdt

            def update(self):
                t, dt = bp.share['t'], bp.share['dt']
                V, h, n = self.integral(self.V.value, self.h.value, self.n.value, t, self.input.value, dt=dt)
                self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
                self.V.value = V
                self.h.value = h
                self.n.value = n
                self.input[:] = 0.

        bm.random.seed()
        hh1 = HH(1, method='exp_euler')
        runner1 = bp.DSRunner(hh1, inputs=('input', 2.), monitors=['V', 'h', 'n'])
        runner1.run(100)
        plt.figure()
        plt.plot(runner1.mon.ts, runner1.mon.V, label='V')
        plt.plot(runner1.mon.ts, runner1.mon.h, label='h')
        plt.plot(runner1.mon.ts, runner1.mon.n, label='n')
        plt.show(block=block)

        hh2 = HH(1, method='exp_euler_auto')
        runner2 = bp.DSRunner(hh2, inputs=('input', 2.), monitors=['V', 'h', 'n'])
        runner2.run(100)
        plt.figure()
        plt.plot(runner2.mon.ts, runner2.mon.V, label='V')
        plt.plot(runner2.mon.ts, runner2.mon.h, label='h')
        plt.plot(runner2.mon.ts, runner2.mon.n, label='n')
        plt.show(block=block)

        diff = (runner2.mon.V - runner1.mon.V).mean()
        self.assertTrue(diff < 1e0)

        plt.close()
