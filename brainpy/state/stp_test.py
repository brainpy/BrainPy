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


import unittest

import brainstate
import brainunit as u
import jax.numpy as jnp
from brainpy.state import STP, STD


class TestSTP(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.in_size = 10
        self.batch_size = 4
        self.dt = 0.1 * u.ms

    def test_init_default_params(self):
        stp = STP(self.in_size)
        self.assertEqual(stp.in_size, (self.in_size,))
        self.assertEqual(stp.out_size, (self.in_size,))
        self.assertEqual(stp.U, 0.15)
        self.assertEqual(stp.tau_f, 1500. * u.ms)
        self.assertEqual(stp.tau_d, 200. * u.ms)

    def test_init_custom_params(self):
        stp = STP(self.in_size, U=0.3, tau_f=500. * u.ms, tau_d=100. * u.ms)
        self.assertEqual(stp.U, 0.3)
        self.assertEqual(stp.tau_f, 500. * u.ms)
        self.assertEqual(stp.tau_d, 100. * u.ms)

    def test_state_init_batched(self):
        stp = STP(self.in_size)
        stp.init_state(self.batch_size)
        self.assertEqual(stp.x.value.shape, (self.batch_size, self.in_size))
        self.assertEqual(stp.u.value.shape, (self.batch_size, self.in_size))
        self.assertTrue(jnp.allclose(stp.x.value, 1.0))
        self.assertTrue(jnp.allclose(stp.u.value, 0.15))

    def test_state_init_unbatched(self):
        stp = STP(self.in_size)
        stp.init_state()
        self.assertEqual(stp.x.value.shape, (self.in_size,))
        self.assertEqual(stp.u.value.shape, (self.in_size,))

    def test_reset_state(self):
        stp = STP(self.in_size)
        stp.init_state(self.batch_size)
        spike = jnp.ones((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            stp.update(spike)
        self.assertFalse(jnp.allclose(stp.x.value, 1.0))
        stp.reset_state(self.batch_size)
        self.assertTrue(jnp.allclose(stp.x.value, 1.0))
        self.assertTrue(jnp.allclose(stp.u.value, 0.15))

    def test_output_zero_no_spike(self):
        stp = STP(self.in_size)
        stp.init_state(self.batch_size)
        no_spike = jnp.zeros((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            out = stp.update(no_spike)
        self.assertEqual(out.shape, (self.batch_size, self.in_size))
        self.assertTrue(jnp.allclose(out, 0.0))

    def test_output_nonzero_with_spike(self):
        stp = STP(self.in_size)
        stp.init_state(self.batch_size)
        spike = jnp.ones((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            out = stp.update(spike)
        self.assertEqual(out.shape, (self.batch_size, self.in_size))
        self.assertTrue(jnp.all(out > 0))

    def test_first_spike_releases_available_resources(self):
        """Regression: STP releases ``u+ . x-`` (resources available *before*
        this spike depletes them), not ``u+ . x+``. From rest x recovers to 1,
        so the released amount equals the post-spike utilization ``u``.
        """
        stp = STP(self.in_size, U=0.15)
        stp.init_state(self.batch_size)
        spike = jnp.ones((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            out = stp.update(spike)
        # x stays at 1.0 (already full), so released == u+ == stp.u.value.
        self.assertTrue(jnp.allclose(out, stp.u.value))
        # The buggy post-depletion output u+.(1-u+) would be strictly smaller.
        self.assertTrue(jnp.all(out > stp.u.value * (1.0 - stp.u.value) + 1e-3))

    def test_u_decay_no_spike(self):
        """Without spikes, u should decay toward 0."""
        stp = STP(self.in_size)
        stp.init_state(self.batch_size)
        u_init = stp.u.value.copy()
        no_spike = jnp.zeros((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            stp.update(no_spike)
        self.assertTrue(jnp.all(stp.u.value < u_init))

    def test_x_recovery_after_depression(self):
        """After a spike depresses x, it should recover without spikes."""
        stp = STP(self.in_size, U=0.3)
        stp.init_state(self.batch_size)
        spike = jnp.ones((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            stp.update(spike)
        x_depressed = stp.x.value.copy()
        self.assertTrue(jnp.all(x_depressed < 1.0))
        no_spike = jnp.zeros((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            stp.update(no_spike)
        self.assertTrue(jnp.all(stp.x.value > x_depressed))

    def test_facilitation_u_increases_on_spike(self):
        """After a spike, u should increase due to facilitation."""
        stp = STP(self.in_size)
        stp.init_state(self.batch_size)
        u_before = stp.u.value.copy()
        spike = jnp.ones((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            stp.update(spike)
        self.assertTrue(jnp.all(stp.u.value > u_before))

    def test_depression_x_decreases_on_spike(self):
        """After a spike, x should decrease due to depression."""
        stp = STP(self.in_size, U=0.3)
        stp.init_state(self.batch_size)
        x_before = stp.x.value.copy()
        spike = jnp.ones((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            stp.update(spike)
        self.assertTrue(jnp.all(stp.x.value < x_before))

    def test_repeated_spikes_depress_resources(self):
        """Continuous spiking should progressively deplete x."""
        stp = STP(self.in_size, U=0.3, tau_d=200. * u.ms)
        stp.init_state(self.batch_size)
        spike = jnp.ones((self.batch_size, self.in_size))
        x_values = []
        with brainstate.environ.context(dt=self.dt):
            for _ in range(20):
                stp.update(spike)
                x_values.append(float(jnp.mean(stp.x.value)))
        for i in range(1, len(x_values)):
            self.assertLessEqual(x_values[i], x_values[i - 1] + 1e-6)

    def test_sparse_spike_pattern(self):
        """Only spiking neurons should be affected."""
        stp = STP(self.in_size)
        stp.init_state(self.batch_size)
        spike = jnp.zeros((self.batch_size, self.in_size))
        spike = spike.at[:, 0].set(1.0)  # only neuron 0 spikes
        with brainstate.environ.context(dt=self.dt):
            out = stp.update(spike)
        self.assertTrue(jnp.all(out[:, 0] > 0))
        self.assertTrue(jnp.all(out[:, 1:] == 0))

    def test_jit_compatible(self):
        stp = STP(self.in_size)
        stp.init_state(self.batch_size)
        call = brainstate.transform.jit(stp)
        spike = jnp.ones((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            out = call(spike)
        self.assertEqual(out.shape, (self.batch_size, self.in_size))
        self.assertTrue(jnp.all(out > 0))

    def test_multidim_input(self):
        in_size = (3, 4)
        stp = STP(in_size)
        stp.init_state(self.batch_size)
        spike = jnp.ones((self.batch_size, *in_size))
        with brainstate.environ.context(dt=self.dt):
            out = stp.update(spike)
        self.assertEqual(out.shape, (self.batch_size, *in_size))

    def test_multiple_steps_jit(self):
        stp = STP(self.in_size)
        stp.init_state(self.batch_size)
        call = brainstate.transform.jit(stp)
        with brainstate.environ.context(dt=self.dt):
            for t in range(50):
                spike = (jnp.ones((self.batch_size, self.in_size))
                         if t % 10 == 0 else
                         jnp.zeros((self.batch_size, self.in_size)))
                out = call(spike)
                self.assertEqual(out.shape, (self.batch_size, self.in_size))


class TestSTD(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.in_size = 10
        self.batch_size = 4
        self.dt = 0.1 * u.ms

    def test_init_default_params(self):
        std = STD(self.in_size)
        self.assertEqual(std.in_size, (self.in_size,))
        self.assertEqual(std.out_size, (self.in_size,))
        self.assertEqual(std.tau, 200. * u.ms)
        self.assertEqual(std.U, 0.07)

    def test_init_custom_params(self):
        std = STD(self.in_size, tau=100. * u.ms, U=0.2)
        self.assertEqual(std.tau, 100. * u.ms)
        self.assertEqual(std.U, 0.2)

    def test_state_init_batched(self):
        std = STD(self.in_size)
        std.init_state(self.batch_size)
        self.assertEqual(std.x.value.shape, (self.batch_size, self.in_size))
        self.assertTrue(jnp.allclose(std.x.value, 1.0))

    def test_state_init_unbatched(self):
        std = STD(self.in_size)
        std.init_state()
        self.assertEqual(std.x.value.shape, (self.in_size,))
        self.assertTrue(jnp.allclose(std.x.value, 1.0))

    def test_reset_state(self):
        std = STD(self.in_size, U=0.2)
        std.init_state(self.batch_size)
        spike = jnp.ones((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            std.update(spike)
        self.assertFalse(jnp.allclose(std.x.value, 1.0))
        std.reset_state(self.batch_size)
        self.assertTrue(jnp.allclose(std.x.value, 1.0))

    def test_output_zero_no_spike(self):
        std = STD(self.in_size)
        std.init_state(self.batch_size)
        no_spike = jnp.zeros((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            out = std.update(no_spike)
        self.assertEqual(out.shape, (self.batch_size, self.in_size))
        self.assertTrue(jnp.allclose(out, 0.0))

    def test_output_nonzero_with_spike(self):
        std = STD(self.in_size)
        std.init_state(self.batch_size)
        spike = jnp.ones((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            out = std.update(spike)
        self.assertEqual(out.shape, (self.batch_size, self.in_size))
        self.assertTrue(jnp.all(out > 0))

    def test_depression_after_spike(self):
        std = STD(self.in_size, U=0.2)
        std.init_state(self.batch_size)
        spike = jnp.ones((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            std.update(spike)
        self.assertTrue(jnp.all(std.x.value < 1.0))

    def test_first_spike_transmits_full_resources(self):
        """Regression: STD output is ``g_syn = x``, the resources available
        *before* this spike depletes them. The first spike from rest is
        therefore undepressed (~1.0), while the state still depresses to 1-U.
        """
        std = STD(self.in_size, U=0.2)
        std.init_state(self.batch_size)
        spike = jnp.ones((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            out = std.update(spike)
        self.assertTrue(jnp.allclose(out, 1.0))
        self.assertTrue(jnp.allclose(std.x.value, 0.8, atol=1e-3))

    def test_recovery_no_spike(self):
        """After depression, x should recover toward 1 without spikes."""
        std = STD(self.in_size, U=0.2)
        std.init_state(self.batch_size)
        spike = jnp.ones((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            std.update(spike)
        x_depressed = std.x.value.copy()
        no_spike = jnp.zeros((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            std.update(no_spike)
        self.assertTrue(jnp.all(std.x.value > x_depressed))

    def test_monotonic_depression_repeated_spikes(self):
        """Continuous spiking should monotonically decrease x."""
        std = STD(self.in_size, U=0.2)
        std.init_state(self.batch_size)
        spike = jnp.ones((self.batch_size, self.in_size))
        x_values = []
        with brainstate.environ.context(dt=self.dt):
            for _ in range(10):
                std.update(spike)
                x_values.append(float(jnp.mean(std.x.value)))
        for i in range(1, len(x_values)):
            self.assertLess(x_values[i], x_values[i - 1])

    def test_sparse_spike_pattern(self):
        """Only spiking neurons should produce output."""
        std = STD(self.in_size)
        std.init_state(self.batch_size)
        spike = jnp.zeros((self.batch_size, self.in_size))
        spike = spike.at[:, 0].set(1.0)
        with brainstate.environ.context(dt=self.dt):
            out = std.update(spike)
        self.assertTrue(jnp.all(out[:, 0] > 0))
        self.assertTrue(jnp.all(out[:, 1:] == 0))

    def test_jit_compatible(self):
        std = STD(self.in_size)
        std.init_state(self.batch_size)
        call = brainstate.transform.jit(std)
        spike = jnp.ones((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            out = call(spike)
        self.assertEqual(out.shape, (self.batch_size, self.in_size))

    def test_multidim_input(self):
        in_size = (3, 4)
        std = STD(in_size)
        std.init_state(self.batch_size)
        spike = jnp.ones((self.batch_size, *in_size))
        with brainstate.environ.context(dt=self.dt):
            out = std.update(spike)
        self.assertEqual(out.shape, (self.batch_size, *in_size))

    def test_x_bounded_between_0_and_1(self):
        """x should remain in [0, 1] after many spikes."""
        std = STD(self.in_size, U=0.3)
        std.init_state(self.batch_size)
        spike = jnp.ones((self.batch_size, self.in_size))
        with brainstate.environ.context(dt=self.dt):
            for _ in range(100):
                std.update(spike)
        self.assertTrue(jnp.all(std.x.value >= 0.0))
        self.assertTrue(jnp.all(std.x.value <= 1.0))


if __name__ == '__main__':
    unittest.main()
