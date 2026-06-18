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
import numpy as np
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm
from brainpy.dyn.neurons import hh


class Test_HH(parameterized.TestCase):
    def test_HH(self):
        model = hh.HH(size=1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'm', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['m'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['n'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['h'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_HH_batching_mode(self):
        model = hh.HH(size=10, mode=bm.batching_mode)
        runner = bp.DSRunner(model,
                             monitors=['V', 'm', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['m'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['n'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['h'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))

    def test_HHLTC(self):
        model = hh.HHLTC(size=1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'm', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['m'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['n'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['h'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_HHLTC_batching_mode(self):
        model = hh.HHLTC(size=10, mode=bm.batching_mode)
        runner = bp.DSRunner(model,
                             monitors=['V', 'm', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['m'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['n'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['h'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))

    def test_MorrisLecar(self):
        model = hh.MorrisLecar(size=1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'W', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['W'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_MorrisLecar_batching_mode(self):
        model = hh.MorrisLecar(size=10, mode=bm.batching_mode)
        runner = bp.DSRunner(model,
                             monitors=['V', 'W', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['W'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))

    def test_MorrisLecarLTC(self):
        model = hh.MorrisLecarLTC(size=1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'W', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['W'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_MorrisLecarLTC_batching_mode(self):
        model = hh.MorrisLecarLTC(size=10, mode=bm.batching_mode)
        runner = bp.DSRunner(model,
                             monitors=['V', 'W', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['W'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))

    def test_WangBuzsakiModel(self):
        model = hh.WangBuzsakiHH(size=1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['n'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['h'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_WangBuzsakiModel_batching_mode(self):
        model = hh.WangBuzsakiHH(size=10, mode=bm.batching_mode)
        runner = bp.DSRunner(model,
                             monitors=['V', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['n'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['h'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))

    def test_WangBuzsakiModelLTC(self):
        model = hh.WangBuzsakiHHLTC(size=1)
        runner = bp.DSRunner(model,
                             monitors=['V', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['n'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['h'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_WangBuzsakiModelLTC_batching_mode(self):
        model = hh.WangBuzsakiHHLTC(size=10, mode=bm.batching_mode)
        runner = bp.DSRunner(model,
                             monitors=['V', 'n', 'h', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['n'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['h'].shape, (1, 100, 10))
        self.assertTupleEqual(runner.mon['spike'].shape, (1, 100, 10))


class Test_CondNeuGroup_synaptic_scaling(parameterized.TestCase):
    """Regression for P8-H1.

    Synaptic currents (returned by ``current_inputs`` / ``sum_current_inputs``)
    are densities, exactly like channel currents. They must therefore NOT be
    rescaled by the ``1e-3 / A`` factor that converts the *external* injected
    current into a density. ``CondNeuGroupLTC`` already does this correctly;
    ``CondNeuGroup`` used to fold the synaptic current into the pre-scaled
    external input, attenuating it by ``1e-3 / A`` whenever ``A != 1e-3``.
    """

    def _one_step_synaptic_dV(self, cls, A, syn_density):
        neu = cls(1, A=A, IL=bp.dyn.IL(1, g_max=0.0, E=-70.))
        neu.reset_state()
        # a constant synaptic current density (independent of V)
        neu.add_inp_fun('syn', lambda V, init=0.: init + syn_density)
        bp.share.save(t=0., dt=0.1, i=0)
        V0 = float(np.asarray(neu.V.value)[0])
        neu.update(0.)  # no external current
        V1 = float(np.asarray(neu.V.value)[0])
        return V1 - V0

    def test_condneugroup_synaptic_current_scaling(self):
        # A != 1e-3 so that the (1e-3 / A) factor is not the identity.
        A = 1.0
        syn = 10.0
        dt, C = 0.1, 1.0
        dv_ltc = self._one_step_synaptic_dV(hh.CondNeuGroupLTC, A, syn)
        dv_cng = self._one_step_synaptic_dV(hh.CondNeuGroup, A, syn)
        # Both classes must apply the synaptic current as an unscaled density.
        self.assertAlmostEqual(dv_ltc, dt * syn / C, places=4)
        self.assertAlmostEqual(dv_cng, dt * syn / C, places=4)
        self.assertAlmostEqual(dv_cng, dv_ltc, places=5)

    def test_external_input_still_scaled(self):
        # The external injected current must STILL be scaled by 1e-3 / A
        # (this is the conversion from absolute current to current density).
        A = 1.0
        x = 10.0
        dt, C = 0.1, 1.0
        neu = hh.CondNeuGroup(1, A=A, IL=bp.dyn.IL(1, g_max=0.0, E=-70.),
                              input_var=False)
        neu.reset_state()
        bp.share.save(t=0., dt=dt, i=0)
        V0 = float(np.asarray(neu.V.value)[0])
        neu.update(x)
        V1 = float(np.asarray(neu.V.value)[0])
        # the external input is converted to a density via (1e-3 / A); compare
        # against the first-order estimate with a loose tolerance (the actual
        # integrator is higher-order, so a small discrepancy is expected).
        expected = dt * (x * (1e-3 / A)) / C
        self.assertAlmostEqual((V1 - V0) / expected, 1.0, places=2)
