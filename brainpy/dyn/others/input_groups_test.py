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
from brainpy.dyn.others import input


class Test_input_Group(parameterized.TestCase):
    def test_SpikeTimeGroup(self):
        bp.math.random.seed()
        model = input.SpikeTimeGroup(size=2, times=[10, 20, 20, 30], indices=[0, 0, 1, 1])
        runner = bp.DSRunner(model,
                             monitors=['spike'],
                             progress_bar=False)
        runner.run(30.)
        self.assertTupleEqual(runner.mon['spike'].shape, (300, 2))

    def test_PoissonGroup(self):
        bp.math.random.seed()
        model = input.PoissonGroup(size=2, freqs=1000)
        runner = bp.DSRunner(model,
                             monitors=['spike'],
                             progress_bar=False)
        runner.run(30.)
        self.assertTupleEqual(runner.mon['spike'].shape, (300, 2))
