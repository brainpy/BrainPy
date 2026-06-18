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
"""Line-coverage tests for ``brainpy/dyn/projections/align_pre.py``.

The existing ``aligns_test.py`` only drives the *merging* variants
(``FullProjAlignPreSDMg`` / ``FullProjAlignPreDSMg``).  Here we add coverage for
the *non*-merging projections and the shared helpers:

* ``FullProjAlignPreSD``  (synapse -> delay) end-to-end, plus its ``update(x)``
  with both an explicit ``x`` and the default ``None`` (delay-read) branch.
* ``FullProjAlignPreDS``  (delay -> synapse) end-to-end.
* ``FullProjAlignPreSDMg`` / ``FullProjAlignPreDSMg`` (so ``align_pre1_add_bef_update``,
  ``align_pre2_add_bef_update``, ``_AlignPreMg`` are exercised in this module too).
* The ``_AlignPre`` helper directly, including its ``delay is None`` branch.
* The projection property accessors (``pre``/``post``/``syn``/``delay``/``out``).
* A constructor type-check error branch.
"""

import numpy as np
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm
from brainpy.dyn.projections.align_pre import _AlignPre

NEU = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)


class _SDNet(bp.DynSysGroup):
    """E -> (syn -> delay) -> E ; I -> (delay -> syn) -> E, non-merging."""

    def __init__(self, delay=0.5):
        super().__init__()
        self.E = bp.dyn.LifRef(8, **NEU)
        self.I = bp.dyn.LifRef(4, **NEU)
        self.E2E = bp.dyn.FullProjAlignPreSD(
            pre=self.E,
            syn=bp.dyn.Expon(size=8, tau=5.),
            delay=delay,
            comm=bp.dnn.AllToAll(8, 8, weight=0.1),
            out=bp.dyn.COBA(E=0.),
            post=self.E)
        self.I2E = bp.dyn.FullProjAlignPreDS(
            pre=self.I,
            delay=delay,
            syn=bp.dyn.Expon(size=4, tau=10.),
            comm=bp.dnn.AllToAll(4, 8, weight=0.1),
            out=bp.dyn.COBA(E=-80.),
            post=self.E)

    def update(self, inp):
        self.E2E()
        self.I2E()
        self.E(inp)
        self.I(inp)
        return self.E.spike.value


class _MgNet(bp.DynSysGroup):
    """Merging counterpart of :class:`_SDNet`."""

    def __init__(self, delay=0.5):
        super().__init__()
        self.E = bp.dyn.LifRef(8, **NEU)
        self.I = bp.dyn.LifRef(4, **NEU)
        self.E2E = bp.dyn.FullProjAlignPreSDMg(
            pre=self.E,
            syn=bp.dyn.Expon.desc(size=8, tau=5.),
            delay=delay,
            comm=bp.dnn.AllToAll(8, 8, weight=0.1),
            out=bp.dyn.COBA(E=0.),
            post=self.E)
        self.I2E = bp.dyn.FullProjAlignPreDSMg(
            pre=self.I,
            delay=delay,
            syn=bp.dyn.Expon.desc(size=4, tau=10.),
            comm=bp.dnn.AllToAll(4, 8, weight=0.1),
            out=bp.dyn.COBA(E=-80.),
            post=self.E)

    def update(self, inp):
        self.E2E()
        self.I2E()
        self.E(inp)
        self.I(inp)
        return self.E.spike.value


class TestAlignPreNonMerging(parameterized.TestCase):
    def test_sd_ds_run(self):
        bm.random.seed()
        bm.set_dt(0.1)
        net = _SDNet()
        spks = bm.for_loop(lambda i: net.step_run(i, 20.), np.arange(50))
        self.assertTupleEqual(tuple(spks.shape), (50, 8))

        # property accessors
        self.assertIs(net.E2E.pre, net.E)
        self.assertIs(net.E2E.post, net.E)
        self.assertIsNotNone(net.E2E.syn)
        self.assertIsNotNone(net.E2E.delay)
        self.assertIsNotNone(net.E2E.out)
        self.assertIs(net.I2E.pre, net.I)
        self.assertIsNotNone(net.I2E.delay)
        self.assertIsNotNone(net.I2E.out)

    def test_sd_update_explicit_x(self):
        bm.random.seed()
        bm.set_dt(0.1)
        net = _SDNet()
        # explicit x bypasses the delay.at(...) read branch
        out = net.E2E.update(bm.ones(8))
        self.assertEqual(out.shape, ())


class TestAlignPreMerging(parameterized.TestCase):
    def test_mg_run(self):
        bm.random.seed()
        bm.set_dt(0.1)
        net = _MgNet()
        spks = bm.for_loop(lambda i: net.step_run(i, 20.), np.arange(50))
        self.assertTupleEqual(tuple(spks.shape), (50, 8))
        self.assertIsNotNone(net.E2E.syn)
        self.assertIsNotNone(net.I2E.syn)


class TestAlignPreHelpers(parameterized.TestCase):
    def test_align_pre_no_delay_branch(self):
        bm.set_dt(0.1)
        syn = bp.dyn.Expon(size=4, tau=5.)
        ap = _AlignPre(syn, delay=None)
        ap.reset_state()
        out = ap.update(bm.ones(4))   # exercises ``x >> self.syn`` (delay is None)
        self.assertTupleEqual(tuple(out.shape), (4,))

    # NOTE: the ``delay is not None`` branch of ``_AlignPre.update`` (``x >> syn
    # >> delay``) is exercised by the running ``_SDNet`` / ``_MgNet`` above.
    # Driving it on a hand-built ``Delay`` outside a transformed graph leaks a
    # JAX tracer, so it is intentionally not duplicated here.

class TestAlignPreErrors(parameterized.TestCase):
    def test_bad_comm_type_raises(self):
        bm.set_dt(0.1)
        E = bp.dyn.LifRef(8, **NEU)
        with self.assertRaises((NotImplementedError, TypeError, ValueError)):
            bp.dyn.FullProjAlignPreSD(
                pre=E,
                syn=bp.dyn.Expon(size=8, tau=5.),
                delay=0.5,
                comm='not_a_dynamical_system',
                out=bp.dyn.COBA(E=0.),
                post=E)


if __name__ == '__main__':
    parameterized.absltest.main()
