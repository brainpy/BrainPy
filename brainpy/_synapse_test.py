# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
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

import brainunit as u
import jax.numpy as jnp
import pytest

import brainstate
from brainpy import Expon, STP, STD


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


if __name__ == '__main__':
    with brainstate.environ.context(dt=0.1):
        unittest.main()
