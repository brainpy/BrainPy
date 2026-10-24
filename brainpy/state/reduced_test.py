# Copyright 2024 BrainX Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from brainpy.state import FitzHughNagumo, HindmarshRose


class TestFitzHughNagumo(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.01 * u.ms)

    def test_construction(self):
        m = FitzHughNagumo(5)
        self.assertEqual(m.varshape, (5,))

    def test_no_repeat_spike_while_above_threshold(self):
        # B4 regression: FitzHugh-Nagumo has no reset, so a per-step threshold
        # test would fire every step while v stays above threshold. The
        # rising-edge detector must report no spike when the *previous* v was
        # already above threshold.
        with brainstate.environ.context(dt=0.01 * u.ms):
            m = FitzHughNagumo(3)
            brainstate.nn.init_all_states(m)
            m.V.value = jnp.ones(3) * 5.0  # already above V_th (=1)
            spk = m.update(0.5)
            npt.assert_allclose(np.asarray(spk), 0.0, atol=1e-6)

    def test_oscillates_single_spike_per_cycle(self):
        with brainstate.environ.context(dt=0.01 * u.ms):
            m = FitzHughNagumo(1)
            brainstate.nn.init_all_states(m)

            def step(i):
                with brainstate.environ.context(i=i):
                    s = m.update(0.5)
                    return s, m.V.value

            spikes, Vs = brainstate.transform.for_loop(step, np.arange(20000))
            spikes = np.asarray(spikes)[:, 0]
            Vs = np.asarray(Vs)[:, 0]
            self.assertTrue(np.all(np.isfinite(Vs)))
            above = Vs > 1.0
            rising = int(np.sum((~above[:-1]) & (above[1:])))
            n_spk = int(spikes.sum())
            self.assertGreaterEqual(n_spk, 1)            # it actually fires
            self.assertLessEqual(abs(n_spk - rising), 1)  # one spike per upward crossing
            self.assertLess(n_spk, int(above.sum()))      # NOT one spike per step-above


class TestHindmarshRose(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.01 * u.ms)

    def test_construction(self):
        m = HindmarshRose(4)
        self.assertEqual(m.varshape, (4,))

    def test_no_repeat_spike_while_above_threshold(self):
        with brainstate.environ.context(dt=0.01 * u.ms):
            m = HindmarshRose(3)
            brainstate.nn.init_all_states(m)
            m.V.value = jnp.ones(3) * 5.0  # already above V_th (=1)
            spk = m.update(2.0)
            npt.assert_allclose(np.asarray(spk), 0.0, atol=1e-6)

    def test_bursts_and_is_bounded(self):
        with brainstate.environ.context(dt=0.01 * u.ms):
            m = HindmarshRose(1)
            brainstate.nn.init_all_states(m)

            def step(i):
                with brainstate.environ.context(i=i):
                    s = m.update(2.0)
                    return s, m.V.value

            spikes, Vs = brainstate.transform.for_loop(step, np.arange(60000))
            spikes = np.asarray(spikes)[:, 0]
            Vs = np.asarray(Vs)[:, 0]
            self.assertTrue(np.all(np.isfinite(Vs)))
            n_spk = int(spikes.sum())
            self.assertGreaterEqual(n_spk, 2)         # bursting produces multiple spikes
            self.assertLess(n_spk, int((Vs > 1.0).sum()))  # not one spike per step-above


if __name__ == '__main__':
    unittest.main()
