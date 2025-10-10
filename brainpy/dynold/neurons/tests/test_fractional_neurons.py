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
from brainpy.dynold.neurons import fractional_models


class Test_Fractional(parameterized.TestCase):
    def test_FractionalFHR(self):
        bp.math.random.seed()
        model = fractional_models.FractionalFHR(size=1, alpha=0.5)
        runner = bp.DSRunner(model,
                             monitors=['V', 'w', 'y', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['w'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['y'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))

    def test_FractionalIzhikevich(self):
        bp.math.random.seed()
        model = fractional_models.FractionalIzhikevich(size=1, alpha=0.5, num_memory=1000)
        runner = bp.DSRunner(model,
                             monitors=['V', 'u', 'spike'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['V'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['u'].shape, (100, 1))
        self.assertTupleEqual(runner.mon['spike'].shape, (100, 1))
