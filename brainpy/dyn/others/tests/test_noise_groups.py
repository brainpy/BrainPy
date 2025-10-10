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
import pytest
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm

pytest.skip("Skip the test due to the jax 0.5.0 version", allow_module_level=True)


class Test_Noise_Group(parameterized.TestCase):
    def test_OU(self):
        bm.random.seed(1234)
        model = bp.dyn.OUProcess(size=1, mean=0., sigma=0.1)
        runner = bp.DSRunner(model,
                             monitors=['x'],
                             progress_bar=False)
        runner.run(10.)
        self.assertTupleEqual(runner.mon['x'].shape, (100, 1))
        x = runner.mon['x']
        self.assertLessEqual(abs(x.mean()), 0.1)
        self.assertLessEqual(abs(x.std() - 0.1), 0.1)
