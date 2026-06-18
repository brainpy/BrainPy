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

from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm
from brainpy.dyn.rates import populations


class TestRate(TestCase):
    def test_fhn(self):
        bm.random.seed()
        fhn = bp.rates.FHN(10)
        self.assertTrue(fhn.tau is not None)

    def test_ffhn(self):
        bm.random.seed()
        ffhn = bp.rates.FeedbackFHN(size=1)
        self.assertTrue(ffhn.tau is not None)

    def test_qif(self):
        bm.random.seed()
        qif = bp.rates.QIF(size=1)
        self.assertTrue(qif.tau is not None)

    def test_slo(self):
        bm.random.seed()
        slo = bp.rates.StuartLandauOscillator(size=1)
        self.assertTrue(slo.x_ou_tau is not None)

    def test_wcm(self):
        bm.random.seed()
        wcm = bp.rates.WilsonCowanModel(size=1)
        self.assertTrue(wcm.x_ou_tau is not None)

    def test_tlm(self):
        bm.random.seed()
        tlm = bp.rates.ThresholdLinearModel(size=1)
        self.assertTrue(tlm.tau_e is not None)


class TestPopulation(parameterized.TestCase):
    @parameterized.named_parameters(
        {'testcase_name': f'noise_of_{name}', 'neuron': name}
        for name in populations.__all__
    )
    def test_runner(self, neuron):
        bm.random.seed()
        model = getattr(populations, neuron)(size=10)
        runner = bp.DSRunner(model, progress_bar=False)
        runner.run(10.)


class TestShape(parameterized.TestCase):
    def test_FHN_shape(self):
        bm.random.seed()
        model = getattr(populations, 'FHN')(size=10)
        runner = bp.DSRunner(model,
                             monitors=['x'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon.x.shape, (100, 10))

    def test_FFHN_shape(self):
        bm.random.seed()
        model = getattr(populations, 'FeedbackFHN')(size=10)
        runner = bp.DSRunner(model,
                             monitors=['x'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon.x.shape, (100, 10))

    def test_QIF_shape(self):
        bm.random.seed()
        model = getattr(populations, 'QIF')(size=10)
        runner = bp.DSRunner(model,
                             monitors=['x'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon.x.shape, (100, 10))

    def test_SLO_shape(self):
        bm.random.seed()
        model = getattr(populations, 'StuartLandauOscillator')(size=10)
        runner = bp.DSRunner(model,
                             monitors=['x'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon.x.shape, (100, 10))

    def test_TLM_shape(self):
        bm.random.seed()
        model = getattr(populations, 'ThresholdLinearModel')(size=10)
        runner = bp.DSRunner(model,
                             monitors=['e'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon.e.shape, (100, 10))
