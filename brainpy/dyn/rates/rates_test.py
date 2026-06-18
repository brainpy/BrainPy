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

import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm
from brainpy.context import share
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


class TestThresholdLinearModelNoise(TestCase):
    """P10-M1: noise must follow Euler-Maruyama ``sqrt(dt)`` scaling."""

    @staticmethod
    def _noise_increment_std(dt):
        # Drive a fresh model with no drift (beta_e=0, tau_e=1) from e=0 so that one
        # step gives e = max(noise_e/tau_e * sqrt(dt) * randn, 0). Measure the std of
        # the (clamped) increment; the positive-half std is proportional to the
        # increment std, so its ratio across dt isolates the dt scaling.
        bm.random.seed(0)
        bm.set_dt(dt)
        m = bp.rates.ThresholdLinearModel(20000, noise_e=1.0, beta_e=0.0, tau_e=1.0)
        m.reset_state()
        share.save(t=0.0, dt=dt, i=0)
        out = np.asarray(m.update(inp_e=0.0))
        pos = out[out > 0]
        return float(pos.std())

    def test_threshold_linear_model_noise_scales_as_sqrt_dt(self):
        s_small = self._noise_increment_std(0.01)
        s_large = self._noise_increment_std(0.1)
        ratio = s_large / s_small
        # sqrt(dt): ratio ~ sqrt(0.1/0.01) = sqrt(10) ~ 3.162.
        # The buggy dt scaling gives ratio ~ 10.
        self.assertAlmostEqual(ratio, np.sqrt(10.0), delta=0.2)

    def test_threshold_linear_model_noise_finite(self):
        bm.random.seed(0)
        bm.set_dt(0.1)
        m = bp.rates.ThresholdLinearModel(8, noise_e=1.0, noise_i=0.5)
        m.reset_state()
        share.save(t=0.0, dt=0.1, i=0)
        out = jnp.asarray(m.update(inp_e=1.0, inp_i=1.0))
        self.assertEqual(out.shape, (8,))
        self.assertTrue(bool(jnp.isfinite(out).all()))


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
