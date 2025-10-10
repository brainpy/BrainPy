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
import pytest

pytest.skip('Test cannot pass in github action.', allow_module_level=True)
import unittest

import brainpy as bp
import brainpy.math as bm
import matplotlib.pyplot as plt

block = False


class FitzHughNagumoModel(bp.DynamicalSystem):
    def __init__(self, method='exp_auto'):
        super(FitzHughNagumoModel, self).__init__()

        # parameters
        self.a = 0.7
        self.b = 0.8
        self.tau = 12.5

        # variables
        self.V = bm.Variable(bm.zeros(1))
        self.w = bm.Variable(bm.zeros(1))
        self.Iext = bm.Variable(bm.zeros(1))

        # functions
        def dV(V, t, w, Iext=0.):
            dV = V - V * V * V / 3 - w + Iext
            return dV

        def dw(w, t, V, a=0.7, b=0.8):
            dw = (V + a - b * w) / self.tau
            return dw

        self.int_V = bp.odeint(dV, method=method)
        self.int_w = bp.odeint(dw, method=method)

    def update(self, tdi):
        t, dt = tdi['t'], tdi['dt']
        self.V.value = self.int_V(self.V, t, self.w, self.Iext, dt)
        self.w.value = self.int_w(self.w, t, self.V, self.a, self.b, dt)
        self.Iext[:] = 0.


class TestBifurcation1D(unittest.TestCase):
    def test_bifurcation_1d(self):
        bp.math.enable_x64()

        @bp.odeint
        def int_x(x, t, a=1., b=1.):
            return bp.math.sin(a * x) + bp.math.cos(b * x)

        pp = bp.analysis.PhasePlane1D(
            model=int_x,
            target_vars={'x': [-bp.math.pi, bp.math.pi]},
            resolutions=0.1
        )
        pp.plot_vector_field()
        pp.plot_fixed_point(show=True)

        bf = bp.analysis.Bifurcation1D(
            model=int_x,
            target_vars={'x': [-bp.math.pi, bp.math.pi]},
            target_pars={'a': [0.5, 1.5], 'b': [0.5, 1.5]},
            resolutions={'a': 0.1, 'b': 0.1}
        )
        bf.plot_bifurcation(show=False)
        plt.show(block=block)
        plt.close()
        bp.math.disable_x64()

    def test_bifurcation_2d(self):
        bp.math.enable_x64()

        model = FitzHughNagumoModel()
        bif = bp.analysis.Bifurcation2D(
            model=model,
            target_vars={'V': [-3., 3.], 'w': [-1, 3.]},
            target_pars={'Iext': [0., 1.]},
            resolutions={'Iext': 0.1}
        )
        bif.plot_bifurcation()
        bif.plot_limit_cycle_by_sim()
        plt.show(block=block)

    #   bp.math.disable_x64()
