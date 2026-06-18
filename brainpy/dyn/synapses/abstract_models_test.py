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

show = False


class TestDualExpon(unittest.TestCase):
    def test_dual_expon(self):
        bm.set(dt=0.01)

        class Net(bp.DynSysGroup):
            def __init__(self, tau_r, tau_d, n_spk):
                super().__init__()

                self.inp = bp.dyn.SpikeTimeGroup(1, bm.zeros(n_spk, dtype=int), bm.linspace(2., 100., n_spk))
                self.proj = bp.dyn.DualExpon(1, tau_rise=tau_r, tau_decay=tau_d)

            def update(self):
                self.proj(self.inp())
                return self.proj.h.value, self.proj.g.value

        for tau_r, tau_d in [(1., 10.), (10., 100.)]:
            for n_spk in [1, 10, 100]:
                net = Net(tau_r, tau_d, n_spk)
                indices = bm.as_numpy(bm.arange(1000))
                hs, gs = bm.for_loop(net.step_run, indices, progress_bar=True)

                bp.visualize.line_plot(indices * bm.get_dt(), hs, legend='h')
                bp.visualize.line_plot(indices * bm.get_dt(), gs, legend='g', show=show)
        plt.close('all')

    def test_dual_expon_v2(self):
        class Net(bp.DynSysGroup):
            def __init__(self, tau_r, tau_d, n_spk):
                super().__init__()

                self.inp = bp.dyn.SpikeTimeGroup(1, bm.zeros(n_spk, dtype=int), bm.linspace(2., 100., n_spk))
                self.syn = bp.dyn.DualExponV2(1, tau_rise=tau_r, tau_decay=tau_d)

            def update(self):
                return self.syn(self.inp())

        for tau_r, tau_d in [(1., 10.), (5., 50.), (10., 100.)]:
            for n_spk in [1, 10, 100]:
                net = Net(tau_r, tau_d, n_spk)
                indices = bm.as_numpy(bm.arange(1000))
                gs = bm.for_loop(net.step_run, indices, progress_bar=True)

                bp.visualize.line_plot(indices * bm.get_dt(), gs, legend='g', show=show)

        plt.close('all')


class TestAlpha(unittest.TestCase):

    def test_v1(self):
        class Net(bp.DynSysGroup):
            def __init__(self, tau, n_spk):
                super().__init__()

                self.inp = bp.dyn.SpikeTimeGroup(1, bm.zeros(n_spk, dtype=int), bm.linspace(2., 100., n_spk))
                self.neu = bp.dyn.LifRef(1)
                self.proj = bp.dyn.FullProjAlignPreDS(self.inp, None,
                                                      bp.dyn.Alpha(1, tau_decay=tau),
                                                      bp.dnn.AllToAll(1, 1, 1.),
                                                      bp.dyn.CUBA(), self.neu)

            def update(self):
                self.inp()
                self.proj()
                self.neu()
                return self.proj.syn.h.value, self.proj.syn.g.value

        for tau in [10.]:
            for n_spk in [1, 10, 50]:
                net = Net(tau=tau, n_spk=n_spk)
                indices = bm.as_numpy(bm.arange(1000))
                hs, gs = bm.for_loop(net.step_run, indices, progress_bar=True)

                bp.visualize.line_plot(indices * bm.get_dt(), hs, legend='h')
                bp.visualize.line_plot(indices * bm.get_dt(), gs, legend='g', show=show)

        plt.close('all')


class TestDualExpEqualTau(unittest.TestCase):
    """Regression tests for P9-H1 / P9-H2: ``tau_rise == tau_decay``."""

    def test_equal_tau_no_crash(self):
        # P9-H1: scalar equal taus previously raised ZeroDivisionError at construction.
        syn = bp.dyn.DualExpon(2, tau_rise=10., tau_decay=10.)
        self.assertTrue(np.all(np.isfinite(np.asarray(syn.a))))
        # array taus with one equal pair previously yielded a NaN coefficient.
        syn2 = bp.dyn.DualExpon(2,
                                tau_rise=bm.asarray([10., 5.]),
                                tau_decay=bm.asarray([10., 50.]))
        self.assertTrue(np.all(np.isfinite(np.asarray(syn2.a))))

    def test_equal_tau_matches_alpha(self):
        # P9-H1: the equal-tau dual-exponential is the normalized alpha function;
        # a single unit spike drives a response that peaks at ~1.0.
        bm.set(dt=0.05)

        class Net(bp.DynSysGroup):
            def __init__(self, tau):
                super().__init__()
                self.inp = bp.dyn.SpikeTimeGroup(1, bm.zeros(1, dtype=int), bm.asarray([1.]))
                self.syn = bp.dyn.DualExpon(1, tau_rise=tau, tau_decay=tau)

            def update(self):
                return self.syn(self.inp())

        net = Net(10.)
        indices = bm.as_numpy(bm.arange(4000))
        gs = bm.for_loop(net.step_run, indices)
        peak = float(np.max(np.asarray(gs)))
        self.assertTrue(np.isfinite(peak))
        self.assertAlmostEqual(peak, 1.0, places=2)
        bm.set(dt=0.1)

    def test_v2_equal_tau_raises(self):
        # P9-H2: DualExponV2 is structurally singular for equal taus; the auto
        # normalizer must raise a clear error rather than crash / output zeros.
        with self.assertRaises(ValueError):
            bp.dyn.DualExponV2(2, tau_rise=10., tau_decay=10.)


class TestSTPReset(unittest.TestCase):
    """Regression tests for P9-H3: per-neuron array ``U``."""

    def test_array_U_reset(self):
        # P9-H3: array U previously crashed in reset_state via Variable.fill_.
        U = bm.asarray([0.1, 0.2, 0.3])
        syn = bp.dyn.STP(3, U=U)
        np.testing.assert_allclose(np.asarray(syn.u.value), np.asarray(U))
        np.testing.assert_allclose(np.asarray(syn.x.value), np.ones(3))

    def test_array_U_run(self):
        # The synapse must also integrate without error for heterogeneous U.
        from brainpy.context import share
        bm.set(dt=0.1)
        syn = bp.dyn.STP(3, U=bm.asarray([0.1, 0.2, 0.3]))
        out = []
        for i in range(20):
            share.save(t=i * 0.1, dt=0.1, i=i)
            spk = bm.asarray([1., 0., 1.]) if i == 5 else bm.asarray([0., 0., 0.])
            out.append(np.asarray(syn.update(spk)))
        out = np.asarray(out)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_scalar_U_batched(self):
        # Scalar U with batching must keep working (no regression).
        syn = bp.dyn.STP(3, U=0.15, mode=bm.BatchingMode(4))
        self.assertEqual(syn.u.shape, (4, 3))
        np.testing.assert_allclose(np.asarray(syn.u.value), np.full((4, 3), 0.15))
