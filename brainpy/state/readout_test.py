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

import brainpy
import brainstate
import brainunit as u
import jax.numpy as jnp


class TestReadoutModels(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.in_size = 3
        self.out_size = 3
        self.batch_size = 4
        self.tau = 5.0
        self.V_th = 1.0
        self.x = jnp.ones((self.batch_size, self.in_size))

    def test_LeakyRateReadout(self):
        with brainstate.environ.context(dt=0.1):
            model = brainpy.state.LeakyRateReadout(in_size=self.in_size, out_size=self.out_size, tau=self.tau)
            model.init_state(batch_size=self.batch_size)
            output = model.update(self.x)
            self.assertEqual(output.shape, (self.batch_size, self.out_size))

    def test_LeakyRateReadout_init_params(self):
        with brainstate.environ.context(dt=0.1):
            model = brainpy.state.LeakyRateReadout(in_size=5, out_size=3, tau=10.0)
            self.assertEqual(model.in_size, (5,))
            self.assertEqual(model.out_size, (3,))

    def test_LeakyRateReadout_weight_shape(self):
        with brainstate.environ.context(dt=0.1):
            model = brainpy.state.LeakyRateReadout(in_size=8, out_size=4, tau=5.0)
            self.assertEqual(model.weight.value.shape, (8, 4))

    def test_LeakyRateReadout_decay_factor(self):
        """Decay factor should be exp(-dt/tau), between 0 and 1."""
        with brainstate.environ.context(dt=0.1):
            model = brainpy.state.LeakyRateReadout(in_size=5, out_size=3, tau=5.0)
            decay = model.decay
            self.assertTrue(jnp.all(decay > 0))
            self.assertTrue(jnp.all(decay < 1))

    def test_LeakyRateReadout_per_unit_tau(self):
        """Per-unit tau is sized to out_size (the readout state dimension),
        so it works when in_size != out_size."""
        with brainstate.environ.context(dt=0.1):
            model = brainpy.state.LeakyRateReadout(
                in_size=5, out_size=3, tau=jnp.array([2., 3., 4.])
            )
            self.assertEqual(model.decay.shape, (3,))
            model.init_state(batch_size=self.batch_size)
            out = model.update(jnp.ones((self.batch_size, 5)))
            self.assertEqual(out.shape, (self.batch_size, 3))

    def test_LeakyRateReadout_accumulation(self):
        """With constant input, output should grow toward a steady state."""
        with brainstate.environ.context(dt=0.1):
            model = brainpy.state.LeakyRateReadout(
                in_size=self.in_size, out_size=self.out_size, tau=self.tau
            )
            model.init_state(batch_size=self.batch_size)
            outputs = []
            for _ in range(20):
                out = model.update(self.x)
                outputs.append(jnp.mean(jnp.abs(out)).item())
            # Output magnitude should increase (accumulation)
            self.assertGreater(outputs[-1], outputs[0])

    def test_LeakyRateReadout_decay_no_input(self):
        """Without input, output should decay toward zero."""
        with brainstate.environ.context(dt=0.1):
            model = brainpy.state.LeakyRateReadout(
                in_size=self.in_size, out_size=self.out_size, tau=self.tau
            )
            model.init_state(batch_size=self.batch_size)
            # First give some input
            model.update(self.x)
            r_after_input = jnp.mean(jnp.abs(model.r.value)).item()
            # Then give zero input
            zero_x = jnp.zeros_like(self.x)
            model.update(zero_x)
            r_after_zero = jnp.mean(jnp.abs(model.r.value)).item()
            self.assertLess(r_after_zero, r_after_input)

    def test_LeakyRateReadout_reset_state(self):
        with brainstate.environ.context(dt=0.1):
            model = brainpy.state.LeakyRateReadout(
                in_size=self.in_size, out_size=self.out_size, tau=self.tau
            )
            model.init_state(batch_size=self.batch_size)
            model.update(self.x)
            self.assertFalse(jnp.allclose(model.r.value, 0.0))
            model.reset_state(batch_size=self.batch_size)
            self.assertTrue(jnp.allclose(model.r.value, 0.0))

    def test_LeakyRateReadout_jit(self):
        with brainstate.environ.context(dt=0.1):
            model = brainpy.state.LeakyRateReadout(
                in_size=self.in_size, out_size=self.out_size, tau=self.tau
            )
            model.init_state(batch_size=self.batch_size)
            call = brainstate.transform.jit(model)
            output = call(self.x)
            self.assertEqual(output.shape, (self.batch_size, self.out_size))

    def test_LeakyRateReadout_multiple_steps_jit(self):
        with brainstate.environ.context(dt=0.1):
            model = brainpy.state.LeakyRateReadout(
                in_size=self.in_size, out_size=self.out_size, tau=self.tau
            )
            model.init_state(batch_size=self.batch_size)
            call = brainstate.transform.jit(model)
            for _ in range(50):
                out = call(self.x)
                self.assertEqual(out.shape, (self.batch_size, self.out_size))


if __name__ == '__main__':
    with brainstate.environ.context(dt=0.1):
        unittest.main()
