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

import numpy as np

import brainpy as bp
import brainpy.math as bm

# Capture the default ``dt`` before any test mutates it. ``DSRunner(dt=...)``
# permanently writes ``dt`` into the global brainstate environment (via
# ``share.save(dt=...)``), which otherwise leaks ``dt`` into later test files
# (e.g. delay tests that assume the default ``dt=0.1``).
_DEFAULT_DT = bm.get_dt()


class _DtRestoreMixin:
    """Restore the global ``dt`` after each test that runs a ``DSRunner``."""

    def tearDown(self):
        bm.set_dt(_DEFAULT_DT)


class TestDSRunner(_DtRestoreMixin, unittest.TestCase):
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


class TestMemoryEfficient(_DtRestoreMixin, unittest.TestCase):
    """Regression tests for ``DSRunner(memory_efficient=True)`` (P14-C1).

    The memory-efficient path collects monitors via a host-side callback and
    stacks the model's ``update()`` outputs manually. A previous bug used
    ``list.append`` inside ``tree_map`` (which returns ``None``) so ``.run()``
    silently returned ``None`` instead of the time-stacked outputs.
    """

    def _scalar_ds(self):
        class ExampleDS(bp.DynamicalSystem):
            def __init__(self):
                super().__init__()
                self.i = bm.Variable(bm.zeros(1))

            def update(self):
                self.i += 1.
                return self.i.value

        return ExampleDS

    def test_output_matches_normal_scalar(self):
        DS = self._scalar_ds()

        out_normal = bp.DSRunner(DS(), dt=1., progress_bar=False,
                                 memory_efficient=False).run(5.)
        out_mem = bp.DSRunner(DS(), dt=1., progress_bar=False,
                              memory_efficient=True).run(5.)

        out_normal = np.asarray(out_normal)
        out_mem = np.asarray(out_mem)
        # the memory-efficient output must not be lost
        self.assertIsNotNone(out_mem.dtype)
        self.assertEqual(out_normal.shape, out_mem.shape)
        self.assertTrue(np.allclose(out_normal, out_mem))
        self.assertTrue(np.allclose(out_mem.ravel(), [1., 2., 3., 4., 5.]))

    def test_output_matches_normal_pytree(self):
        # the update returns a dict (a non-trivial pytree of outputs)
        class ExampleDS(bp.DynamicalSystem):
            def __init__(self):
                super().__init__()
                self.i = bm.Variable(bm.zeros(2))

            def update(self):
                self.i += 1.
                return {'a': self.i.value, 'b': self.i.value * 2.}

        out_normal = bp.DSRunner(ExampleDS(), dt=1., progress_bar=False,
                                 memory_efficient=False).run(4.)
        out_mem = bp.DSRunner(ExampleDS(), dt=1., progress_bar=False,
                              memory_efficient=True).run(4.)

        for key in ('a', 'b'):
            a = np.asarray(out_normal[key])
            b = np.asarray(out_mem[key])
            self.assertEqual(a.shape, b.shape)
            self.assertEqual(a.shape, (4, 2))
            self.assertTrue(np.allclose(a, b))

    def test_monitors_still_match(self):
        DS = self._scalar_ds()
        r_n = bp.DSRunner(DS(), dt=1., monitors=['i'], progress_bar=False,
                          memory_efficient=False)
        r_n.run(5.)
        r_m = bp.DSRunner(DS(), dt=1., monitors=['i'], progress_bar=False,
                          memory_efficient=True)
        r_m.run(5.)
        self.assertTrue(np.allclose(np.asarray(r_n.mon['i']),
                                    np.asarray(r_m.mon['i'])))
        self.assertTrue(np.allclose(np.asarray(r_n.mon['ts']),
                                    np.asarray(r_m.mon['ts'])))

    def test_output_none_when_update_returns_none(self):
        # an ``update()`` with no explicit return must give ``None`` in both
        # paths (the all-``None`` pytree collapses to ``None``), not crash.
        class DS(bp.DynamicalSystem):
            def __init__(self):
                super().__init__()
                self.i = bm.Variable(bm.zeros(1))

            def update(self):
                self.i += 1.  # returns None

        out_n = bp.DSRunner(DS(), dt=1., monitors=['i'], progress_bar=False,
                            memory_efficient=False).run(5.)
        r_m = bp.DSRunner(DS(), dt=1., monitors=['i'], progress_bar=False,
                          memory_efficient=True)
        out_m = r_m.run(5.)
        self.assertIsNone(out_n)
        self.assertIsNone(out_m)
        self.assertTrue(np.allclose(np.asarray(r_m.mon['i']).ravel(),
                                    [1., 2., 3., 4., 5.]))

    def test_batched_monitor_axis_matches_normal(self):
        """P14-H1: for BatchingMode + data_first_axis='B' (the default for
        batched models) the memory-efficient monitors must come out batch-major
        ``(B, T, ...)``, identical to the standard path. The bug returned
        time-major ``(T, B, ...)`` only for the memory-efficient path."""

        class Net(bp.DynamicalSystem):
            def __init__(self):
                super().__init__(mode=bm.BatchingMode(4))
                self.n = bp.dyn.LifRef(3, mode=bm.BatchingMode(4))

            def update(self, inp):
                self.n(inp)
                return self.n.V.value

        inp = bm.ones((4, 8, 3)) * 2.0  # (batch, time, features), data_first_axis='B'

        bm.random.seed(0)
        net = Net(); net.reset(4)
        r_n = bp.DSRunner(net, monitors=['n.V'], memory_efficient=False,
                          progress_bar=False)
        out_n = r_n.run(inputs=inp)

        bm.random.seed(0)
        net2 = Net(); net2.reset(4)
        r_m = bp.DSRunner(net2, monitors=['n.V'], memory_efficient=True,
                          progress_bar=False)
        out_m = r_m.run(inputs=inp)

        # outputs are batch-major in both paths
        self.assertEqual(np.asarray(out_n).shape, np.asarray(out_m).shape)
        self.assertEqual(np.asarray(out_m).shape, (4, 8, 3))
        self.assertTrue(np.allclose(np.asarray(out_n), np.asarray(out_m)))

        # monitors must share the same (batch, time, features) layout
        self.assertEqual(np.asarray(r_n.mon['n.V']).shape,
                         np.asarray(r_m.mon['n.V']).shape)
        self.assertEqual(np.asarray(r_m.mon['n.V']).shape, (4, 8, 3))
        self.assertTrue(np.allclose(np.asarray(r_n.mon['n.V']),
                                    np.asarray(r_m.mon['n.V'])))
