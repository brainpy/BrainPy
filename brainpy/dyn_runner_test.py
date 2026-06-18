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


class TestDSRunner(unittest.TestCase):
    def test1(self):
        class ExampleDS(bp.DynamicalSystem):
            def __init__(self):
                super(ExampleDS, self).__init__()
                self.i = bm.Variable(bm.zeros(1))

            def update(self):
                self.i += 1

        ds = ExampleDS()
        runner = bp.DSRunner(ds, dt=1., monitors=['i'], progress_bar=False)
        runner.run(100.)

    def test_t_and_dt(self):
        class ExampleDS(bp.DynamicalSystem):
            def __init__(self):
                super(ExampleDS, self).__init__()
                self.i = bm.Variable(bm.zeros(1))

            def update(self):
                self.i += 1 * bp.share['dt']

        runner = bp.DSRunner(ExampleDS(), dt=1., monitors=['i'], progress_bar=False)
        runner.run(100.)

    def test_DSView(self):
        class EINet(bp.Network):
            def __init__(self, scale=1.0, method='exp_auto'):
                super(EINet, self).__init__()

                # network size
                num_exc = int(800 * scale)
                num_inh = int(200 * scale)

                # neurons
                pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)
                self.E = bp.neurons.LIF(num_exc, **pars, method=method)
                self.I = bp.neurons.LIF(num_inh, **pars, method=method)
                self.E.V[:] = bm.random.randn(num_exc) * 2 - 55.
                self.I.V[:] = bm.random.randn(num_inh) * 2 - 55.

                # synapses
                we = 0.6 / scale  # excitatory synaptic weight (voltage)
                wi = 6.7 / scale  # inhibitory synaptic weight
                self.E2E = bp.synapses.Exponential(self.E, self.E[:100], bp.conn.FixedProb(0.02),
                                                   output=bp.synouts.COBA(E=0.), g_max=we,
                                                   tau=5., method=method)
                self.E2I = bp.synapses.Exponential(self.E, self.I[:100], bp.conn.FixedProb(0.02),
                                                   output=bp.synouts.COBA(E=0.), g_max=we,
                                                   tau=5., method=method)
                self.I2E = bp.synapses.Exponential(self.I, self.E[:100], bp.conn.FixedProb(0.02),
                                                   output=bp.synouts.COBA(E=-80.), g_max=wi,
                                                   tau=10., method=method)
                self.I2I = bp.synapses.Exponential(self.I, self.I[:100], bp.conn.FixedProb(0.02),
                                                   output=bp.synouts.COBA(E=-80.), g_max=wi,
                                                   tau=10., method=method)

        bm.random.seed()

        net = EINet(scale=1., method='exp_auto')
        # with JIT
        runner = bp.DSRunner(net, monitors={'E.spike': net.E.spike},
                             inputs=[(net.E.input, 20.), (net.I.input, 20.)]).run(1.)

        # without JIT
        runner = bp.DSRunner(net, monitors={'E.spike': net.E.spike},
                             inputs=[(net.E.input, 20.), (net.I.input, 20.)], jit=False).run(0.2)


class TestMemoryEfficient(unittest.TestCase):
    pass
