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
from brainpy.dyn.others import input


class Test_input(parameterized.TestCase):
    def test_SpikeTimeGroup(self):
        model = input.SpikeTimeGroup(size=2, times=[10, 20, 20, 30], indices=[0, 0, 1, 1])
        runner = bp.DSRunner(model,
                             monitors=['spike'],
                             progress_bar=False)
        runner.run(30.)
        self.assertTupleEqual(runner.mon['spike'].shape, (300, 2))

    def test_PoissonGroup(self):
        model = input.PoissonGroup(size=2, freqs=1000)
        runner = bp.DSRunner(model,
                             monitors=['spike'],
                             progress_bar=False)
        runner.run(30.)
        self.assertTupleEqual(runner.mon['spike'].shape, (300, 2))

    def test_PoissonGroup_fires_at_expected_rate(self):
        # Regression: PoissonGroup must actually emit spikes at ~freqs Hz.
        # A dtype bug (boolean rand_like) previously made it fire at 0 Hz while
        # still returning the correct output shape, so a shape-only check missed it.
        bm.random.seed(1234)
        freqs = 200.  # Hz
        num = 200
        duration = 1000.  # ms
        model = input.PoissonGroup(size=num, freqs=freqs)
        runner = bp.DSRunner(model, monitors=['spike'], progress_bar=False)
        runner.run(duration)
        spikes = np.asarray(runner.mon['spike'])
        empirical_rate = spikes.sum() / num / (duration / 1000.)
        # Poisson counting noise here is < 1%; 15% tolerance is safe yet still
        # rejects the 0 Hz regression by a wide margin.
        self.assertAlmostEqual(empirical_rate, freqs, delta=0.15 * freqs)

    def test_PoissonGroup_fires_in_batching_mode(self):
        # The float draw must preserve the batched shape of ``spike`` and still
        # produce the expected rate across the batch axis.
        bm.random.seed(1234)
        freqs = 200.  # Hz
        num, batch = 30, 4
        duration = 1000.  # ms
        model = input.PoissonGroup(size=num, freqs=freqs, mode=bm.BatchingMode(batch))
        runner = bp.DSRunner(model, monitors=['spike'], progress_bar=False)
        runner.run(duration)
        spikes = np.asarray(runner.mon['spike'])  # (batch, time, num)
        self.assertEqual(spikes.shape[0], batch)
        self.assertEqual(spikes.shape[-1], num)
        empirical_rate = spikes.sum() / (batch * num) / (duration / 1000.)
        self.assertAlmostEqual(empirical_rate, freqs, delta=0.15 * freqs)
