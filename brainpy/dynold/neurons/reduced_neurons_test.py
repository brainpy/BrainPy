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
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm
from brainpy.dynold.neurons import reduced_models


class Test_Reduced(parameterized.TestCase):
    @parameterized.named_parameters(
        {'testcase_name': f'noise_of_{name}', 'neuron': name}
        for name in reduced_models.__all__
    )
    def test_run_shape(self, neuron):
        bm.random.seed()
        model = getattr(reduced_models, neuron)(size=1)
        if neuron == 'LeakyIntegrator':
            runner = bp.DSRunner(model,
                                 monitors=['V'],
                                 progress_bar=False)
            runner.run(10.)
            self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        else:
            runner = bp.DSRunner(model,
                                 monitors=['V', 'spike'],
                                 progress_bar=False)
            runner.run(10.)
            self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
            self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    @parameterized.named_parameters(
        {'testcase_name': f'noise_of_{name}', 'neuron': name}
        for name in reduced_models.__all__
    )
    def test_noise_shape(self, neuron):
        bm.random.seed()
        model = getattr(reduced_models, neuron)(size=1, noise=0.1)
        if neuron == 'LeakyIntegrator':
            runner = bp.DSRunner(model,
                                 monitors=['V'],
                                 progress_bar=False)
            runner.run(10.)
            self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        else:
            runner = bp.DSRunner(model,
                                 monitors=['V', 'spike'],
                                 progress_bar=False)
            runner.run(10.)
            self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
            self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    @parameterized.named_parameters(
        {'testcase_name': f'noise_of_{name}', 'neuron': name}
        for name in reduced_models.__all__
    )
    def test_training_shape(self, neuron):
        bm.random.seed()
        if neuron == 'FHN':
            model = getattr(reduced_models, neuron)(size=10)
            runner = bp.DSRunner(model,
                                 monitors=['V'],
                                 progress_bar=False)
            runner.run(10.)
            self.assertTupleEqual(runner.mon['V'].shape, (100, 10))
        else:
            model = getattr(reduced_models, neuron)(size=10, mode=bm.training_mode)
            runner = bp.DSRunner(model,
                                 monitors=['V'],
                                 progress_bar=False)
            runner.run(10.)
            self.assertTupleEqual(runner.mon['V'].shape, (1, 100, 10))


class TestBellecAdaptation(parameterized.TestCase):
    """P11-M2 regression: the SFA adaptation variable ``a`` must start at rest.

    The threshold adaptation contributes ``beta * a`` to the effective firing
    threshold (``V_th + beta * a``). The historical default ``OneInit(-50.)``
    started ``a`` deeply negative, dropping the effective threshold by tens of
    mV for thousands of ms and making a cold-started neuron fire spuriously.
    The default must be a rest value (~0).
    """

    @parameterized.named_parameters(
        {'testcase_name': 'ALIFBellec2020', 'neuron': 'ALIFBellec2020'},
        {'testcase_name': 'LIF_SFA_Bellec2020', 'neuron': 'LIF_SFA_Bellec2020'},
    )
    def test_default_adaptation_starts_at_rest(self, neuron):
        bm.random.seed(0)
        model = getattr(reduced_models, neuron)(size=4)
        model.reset_state()
        a0 = bm.as_jax(model.a.value)
        # adaptation starts at zero (no spurious sub-threshold offset)
        self.assertTrue(bool(bm.all(a0 == 0.)))
