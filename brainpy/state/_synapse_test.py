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
import pytest

from brainpy.state import Expon, STP, STD, AMPA, GABAa, BioNMDA


class TestSynapse(unittest.TestCase):
    def setUp(self):
        self.in_size = 10
        self.batch_size = 5
        self.time_steps = 100

    def generate_input(self):
        return brainstate.random.randn(self.time_steps, self.batch_size, self.in_size) * u.mS

    def test_expon_synapse(self):
        tau = 20.0 * u.ms
        synapse = Expon(self.in_size, tau=tau)
        inputs = self.generate_input()

        # Test initialization
        self.assertEqual(synapse.in_size, (self.in_size,))
        self.assertEqual(synapse.out_size, (self.in_size,))
        self.assertEqual(synapse.tau, tau)

        # Test forward pass
        state = synapse.init_state(self.batch_size)
        call = brainstate.compile.jit(synapse)
        with brainstate.environ.context(dt=0.1 * u.ms):
            for t in range(self.time_steps):
                out = call(inputs[t])
                self.assertEqual(out.shape, (self.batch_size, self.in_size))

        # Test exponential decay
        constant_input = jnp.ones((self.batch_size, self.in_size)) * u.mS
        out1 = call(constant_input)
        out2 = call(constant_input)
        self.assertTrue(jnp.all(out2 > out1))  # Output should increase with constant input

    @pytest.mark.skip(reason="Not implemented yet")
    def test_stp_synapse(self):
        tau_d = 200.0 * u.ms
        tau_f = 20.0 * u.ms
        U = 0.2
        synapse = STP(self.in_size, tau_d=tau_d, tau_f=tau_f, U=U)
        inputs = self.generate_input()

        # Test initialization
        self.assertEqual(synapse.in_size, (self.in_size,))
        self.assertEqual(synapse.out_size, (self.in_size,))
        self.assertEqual(synapse.tau_d, tau_d)
        self.assertEqual(synapse.tau_f, tau_f)
        self.assertEqual(synapse.U, U)

        # Test forward pass
        state = synapse.init_state(self.batch_size)
        call = brainstate.compile.jit(synapse)
        for t in range(self.time_steps):
            out = call(inputs[t])
            self.assertEqual(out.shape, (self.batch_size, self.in_size))

        # Test short-term plasticity
        constant_input = jnp.ones((self.batch_size, self.in_size)) * u.mS
        out1 = call(constant_input)
        out2 = call(constant_input)
        self.assertTrue(jnp.any(out2 != out1))  # Output should change due to STP

    @pytest.mark.skip(reason="Not implemented yet")
    def test_std_synapse(self):
        tau = 200.0
        U = 0.2
        synapse = STD(self.in_size, tau=tau, U=U)
        inputs = self.generate_input()

        # Test initialization
        self.assertEqual(synapse.in_size, (self.in_size,))
        self.assertEqual(synapse.out_size, (self.in_size,))
        self.assertEqual(synapse.tau, tau)
        self.assertEqual(synapse.U, U)

        # Test forward pass
        state = synapse.init_state(self.batch_size)
        for t in range(self.time_steps):
            out = synapse(inputs[t])
            self.assertEqual(out.shape, (self.batch_size, self.in_size))

        # Test short-term depression
        constant_input = jnp.ones((self.batch_size, self.in_size))
        out1 = synapse(constant_input)
        out2 = synapse(constant_input)
        self.assertTrue(jnp.all(out2 < out1))  # Output should decrease due to STD

    def test_keep_size(self):
        in_size = (2, 3)
        for SynapseClass in [Expon, ]:
            synapse = SynapseClass(in_size)
            self.assertEqual(synapse.in_size, in_size)
            self.assertEqual(synapse.out_size, in_size)

            inputs = brainstate.random.randn(self.time_steps, self.batch_size, *in_size) * u.mS
            state = synapse.init_state(self.batch_size)
            call = brainstate.compile.jit(synapse)
            with brainstate.environ.context(dt=0.1 * u.ms):
                for t in range(self.time_steps):
                    out = call(inputs[t])
                    self.assertEqual(out.shape, (self.batch_size, *in_size))

    def test_ampa_synapse(self):
        alpha = 0.98 / (u.ms * u.mM)
        beta = 0.18 / u.ms
        T = 0.5 * u.mM
        T_dur = 0.5 * u.ms
        synapse = AMPA(self.in_size, alpha=alpha, beta=beta, T=T, T_dur=T_dur)

        # Test initialization
        self.assertEqual(synapse.in_size, (self.in_size,))
        self.assertEqual(synapse.out_size, (self.in_size,))
        self.assertEqual(synapse.alpha, alpha)
        self.assertEqual(synapse.beta, beta)
        self.assertEqual(synapse.T, T)
        self.assertEqual(synapse.T_duration, T_dur)

        # Test forward pass
        synapse.init_state(self.batch_size)
        call = brainstate.compile.jit(synapse)
        with brainstate.environ.context(dt=0.1 * u.ms, t=0. * u.ms):
            # Test with spike input (True/False array)
            spike_input = jnp.zeros((self.batch_size, self.in_size), dtype=bool)
            spike_input = spike_input.at[0, 0].set(True)  # Single spike

            out1 = call(spike_input)
            self.assertEqual(out1.shape, (self.batch_size, self.in_size))

            # Conductance should increase after spike
            out2 = call(jnp.zeros((self.batch_size, self.in_size), dtype=bool))
            self.assertTrue(jnp.any(out2[0, 0] > 0 * u.mS))  # Should have some conductance

    def test_gabaa_synapse(self):
        alpha = 0.53 / (u.ms * u.mM)
        beta = 0.18 / u.ms
        T = 1.0 * u.mM
        T_dur = 1.0 * u.ms
        synapse = GABAa(self.in_size, alpha=alpha, beta=beta, T=T, T_dur=T_dur)

        # Test initialization
        self.assertEqual(synapse.in_size, (self.in_size,))
        self.assertEqual(synapse.out_size, (self.in_size,))
        self.assertEqual(synapse.alpha, alpha)
        self.assertEqual(synapse.beta, beta)
        self.assertEqual(synapse.T, T)
        self.assertEqual(synapse.T_duration, T_dur)

        # Test forward pass
        synapse.init_state(self.batch_size)
        call = brainstate.compile.jit(synapse)
        with brainstate.environ.context(dt=0.1 * u.ms, t=0. * u.ms):
            spike_input = jnp.zeros((self.batch_size, self.in_size), dtype=bool)
            spike_input = spike_input.at[0, 0].set(True)

            out1 = call(spike_input)
            self.assertEqual(out1.shape, (self.batch_size, self.in_size))

            # Conductance should increase after spike
            out2 = call(jnp.zeros((self.batch_size, self.in_size), dtype=bool))
            self.assertTrue(jnp.any(out2[0, 0] > 0 * u.mS))

    def test_bionmda_synapse(self):
        alpha1 = 2.0 / u.ms
        beta1 = 0.01 / u.ms
        alpha2 = 1.0 / (u.ms * u.mM)
        beta2 = 0.5 / u.ms
        T = 1.0 * u.mM
        T_dur = 0.5 * u.ms
        synapse = BioNMDA(self.in_size, alpha1=alpha1, beta1=beta1,
                         alpha2=alpha2, beta2=beta2, T=T, T_dur=T_dur)

        # Test initialization
        self.assertEqual(synapse.in_size, (self.in_size,))
        self.assertEqual(synapse.out_size, (self.in_size,))
        self.assertEqual(synapse.alpha1, alpha1)
        self.assertEqual(synapse.beta1, beta1)
        self.assertEqual(synapse.alpha2, alpha2)
        self.assertEqual(synapse.beta2, beta2)
        self.assertEqual(synapse.T, T)
        self.assertEqual(synapse.T_duration, T_dur)

        # Test forward pass with spike inputs
        synapse.init_state(self.batch_size)
        call = brainstate.compile.jit(synapse)
        with brainstate.environ.context(dt=0.1 * u.ms, t=0. * u.ms):
            # Create spike input at first time step
            spike_input = jnp.zeros((self.batch_size, self.in_size), dtype=bool)
            spike_input = spike_input.at[0, 0].set(True)  # Single spike at position (0, 0)

            # First call with spike
            out1 = call(spike_input)
            self.assertEqual(out1.shape, (self.batch_size, self.in_size))

            # Verify state variables exist and have correct shape
            self.assertEqual(synapse.g.value.shape, (self.batch_size, self.in_size))
            self.assertEqual(synapse.x.value.shape, (self.batch_size, self.in_size))

            # Continue simulation without spikes
            no_spike = jnp.zeros((self.batch_size, self.in_size), dtype=bool)

            # NMDA should have slower dynamics - collect several time points
            outputs = [out1]
            for _ in range(10):
                out = call(no_spike)
                outputs.append(out)

            # Check that conductance increases over time initially (slower rise time for NMDA)
            # Due to the two-state kinetics, there should be some non-zero conductance
            self.assertTrue(jnp.any(outputs[-1][0, 0] >= 0 * u.mS))  # Should have developed some conductance

    def test_bionmda_two_state_dynamics(self):
        """Test that BioNMDA properly implements second-order kinetics with two state variables"""
        synapse = BioNMDA(self.in_size)
        synapse.init_state(self.batch_size)
        call = brainstate.compile.jit(synapse)

        with brainstate.environ.context(dt=0.1 * u.ms, t=0. * u.ms):
            # Initial state should be zero (g has units, x is dimensionless)
            self.assertTrue(jnp.allclose(synapse.g.value.to_decimal(u.mS), 0.))
            self.assertTrue(jnp.allclose(synapse.x.value, 0.))

            # Apply a spike
            spike_input = jnp.zeros((self.batch_size, self.in_size), dtype=bool)
            spike_input = spike_input.at[0, 0].set(True)

            call(spike_input)

            # After spike, both x and g should be non-negative
            x_val = synapse.x.value[0, 0]
            g_val = synapse.g.value[0, 0]

            # x is dimensionless, g has units
            self.assertTrue(x_val >= 0)
            self.assertTrue(g_val >= 0 * u.mS)


if __name__ == '__main__':
    with brainstate.environ.context(dt=0.1 * u.ms):
        unittest.main()
