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
