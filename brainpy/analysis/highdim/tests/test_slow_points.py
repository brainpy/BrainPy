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

import brainpy as bp
import brainpy.math as bm


class HH(bp.dyn.NeuDyn):
    def __init__(self, size, ENa=50., gNa=120., EK=-77., gK=36., EL=-54.387, gL=0.03,
                 V_th=20., C=1.0, name=None):
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

        # variables
        self.V = bm.Variable(bm.ones(self.num) * -65.)
        self.m = bm.Variable(0.5 * bm.ones(self.num))
        self.h = bm.Variable(0.6 * bm.ones(self.num))
        self.n = bm.Variable(0.32 * bm.ones(self.num))
        self.spike = bm.Variable(bm.zeros(size, dtype=bool))
        self.input = bm.Variable(bm.zeros(size))

        # integral functions
        self.int_h = bp.ode.ExponentialEuler(self.dh)
        self.int_n = bp.ode.ExponentialEuler(self.dn)
        self.int_m = bp.ode.ExponentialEuler(self.dm)
        self.int_V = bp.ode.ExponentialEuler(self.dV)

    def dh(self, h, t, V):
        alpha = 0.07 * bm.exp(-(V + 65) / 20.)
        beta = 1 / (1 + bm.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        return dhdt

    def dn(self, n, t, V):
        alpha = 0.01 * (V + 55) / (1 - bm.exp(-(V + 55) / 10))
        beta = 0.125 * bm.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        return dndt

    def dm(self, m, t, V):
        alpha = 0.1 * (V + 40) / (1 - bm.exp(-(V + 40) / 10))
        beta = 4.0 * bm.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        return dmdt

    def dV(self, V, t, m, h, n, Iext):
        I_Na = (self.gNa * m ** 3.0 * h) * (V - self.ENa)
        I_K = (self.gK * n ** 4.0) * (V - self.EK)
        I_leak = self.gL * (V - self.EL)
        dVdt = (- I_Na - I_K - I_leak + Iext) / self.C
        return dVdt

    def update(self, tdi):
        t, dt = tdi.t, tdi.dt
        m = self.int_m(self.m.value, t, self.V.value, dt=dt)
        h = self.int_h(self.h.value, t, self.V.value, dt=dt)
        n = self.int_n(self.n.value, t, self.V.value, dt=dt)
        V = self.int_V(self.V.value, t, self.m.value, self.h.value, self.n.value, self.input.value, dt=dt)
        self.spike.value = bm.logical_and(self.V < self.V_th, V >= self.V_th)
        self.V.value = V
        self.h.value = h
        self.n.value = n
        self.m.value = m
        self.input[:] = 0.


class TestFixedPointsFinding(unittest.TestCase):
    def test_opt_solver_for_func1(self):
        gamma = 0.641  # Saturation factor for gating variable
        tau = 0.06  # Synaptic time constant [sec]
        a = 270.
        b = 108.
        d = 0.154

        JE = 0.3725  # self-coupling strength [nA]
        JI = -0.1137  # cross-coupling strength [nA]
        JAext = 0.00117  # Stimulus input strength [nA]

        mu = 20.  # Stimulus firing rate [spikes/sec]
        coh = 0.5  # Stimulus coherence [%]
        Ib1 = 0.3297
        Ib2 = 0.3297

        def ds1(s1, t, s2, coh=0.5, mu=20.):
            I1 = JE * s1 + JI * s2 + Ib1 + JAext * mu * (1. + coh)
            r1 = (a * I1 - b) / (1. - bm.exp(-d * (a * I1 - b)))
            return - s1 / tau + (1. - s1) * gamma * r1

        def ds2(s2, t, s1, coh=0.5, mu=20.):
            I2 = JE * s2 + JI * s1 + Ib2 + JAext * mu * (1. - coh)
            r2 = (a * I2 - b) / (1. - bm.exp(-d * (a * I2 - b)))
            return - s2 / tau + (1. - s2) * gamma * r2

        def step(s):
            return bm.asarray([ds1(s[0], 0., s[1]), ds2(s[1], 0., s[0])])

        rng = bm.random.RandomState(123)
        finder = bp.analysis.SlowPointFinder(f_cell=step, f_type=bp.analysis.CONTINUOUS)
        finder.find_fps_with_opt_solver(rng.random((100, 2)))

    def test_opt_solver_for_ds1(self):
        hh = HH(1)
        finder = bp.analysis.SlowPointFinder(f_cell=hh, excluded_vars=[hh.input, hh.spike])
        rng = bm.random.RandomState(123)

        with self.assertRaises(ValueError):
            finder.find_fps_with_opt_solver(rng.random((100, 4)))

        finder.find_fps_with_opt_solver({'V': rng.random((100, 1)),
                                         'm': rng.random((100, 1)),
                                         'h': rng.random((100, 1)),
                                         'n': rng.random((100, 1))})

    def test_gd_method_for_func1(self):
        gamma = 0.641  # Saturation factor for gating variable
        tau = 0.06  # Synaptic time constant [sec]
        a = 270.
        b = 108.
        d = 0.154

        JE = 0.3725  # self-coupling strength [nA]
        JI = -0.1137  # cross-coupling strength [nA]
        JAext = 0.00117  # Stimulus input strength [nA]

        mu = 20.  # Stimulus firing rate [spikes/sec]
        coh = 0.5  # Stimulus coherence [%]
        Ib1 = 0.3297
        Ib2 = 0.3297

        def ds1(s1, t, s2, coh=0.5, mu=20.):
            I1 = JE * s1 + JI * s2 + Ib1 + JAext * mu * (1. + coh)
            r1 = (a * I1 - b) / (1. - bm.exp(-d * (a * I1 - b)))
            return - s1 / tau + (1. - s1) * gamma * r1

        def ds2(s2, t, s1, coh=0.5, mu=20.):
            I2 = JE * s2 + JI * s1 + Ib2 + JAext * mu * (1. - coh)
            r2 = (a * I2 - b) / (1. - bm.exp(-d * (a * I2 - b)))
            return - s2 / tau + (1. - s2) * gamma * r2

        def step(s):
            return bm.asarray([ds1(s[0], 0., s[1]), ds2(s[1], 0., s[0])])

        rng = bm.random.RandomState(123)
        finder = bp.analysis.SlowPointFinder(f_cell=step, f_type=bp.analysis.CONTINUOUS)
        finder.find_fps_with_gd_method(rng.random((100, 2)), num_opt=100)

    def test_gd_method_for_func2(self):
        hh = HH(1)
        finder = bp.analysis.SlowPointFinder(f_cell=hh, excluded_vars=[hh.input, hh.spike])
        rng = bm.random.RandomState(123)

        with self.assertRaises(ValueError):
            finder.find_fps_with_opt_solver(rng.random((100, 4)))

        finder.find_fps_with_gd_method(
            {'V': rng.random((100, 1)),
             'm': rng.random((100, 1)),
             'h': rng.random((100, 1)),
             'n': rng.random((100, 1))},
            num_opt=100
        )
