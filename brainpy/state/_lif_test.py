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
import jax
import jax.numpy as jnp

from brainpy.state import IF, LIF, ALIF


class TestNeuron(unittest.TestCase):
    def setUp(self):
        self.in_size = 10
        self.batch_size = 5
        self.time_steps = 100

    def test_neuron_base_class(self):
        with self.assertRaises(NotImplementedError):
            brainstate.nn.Neuron(self.in_size).get_spike()  # Neuron is an abstract base class

    def generate_input(self):
        return brainstate.random.randn(self.time_steps, self.batch_size, self.in_size) * u.mA

    def test_if_neuron(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            neuron = IF(self.in_size)
            inputs = self.generate_input()

            # Test initialization
            self.assertEqual(neuron.in_size, (self.in_size,))
            self.assertEqual(neuron.out_size, (self.in_size,))

            # Test forward pass
            state = neuron.init_state(self.batch_size)

            for t in range(self.time_steps):
                out = neuron(inputs[t])
                self.assertEqual(out.shape, (self.batch_size, self.in_size))

            # Test spike generation
            v = jnp.linspace(-1, 1, 100) * u.mV
            spikes = neuron.get_spike(v)
            self.assertTrue(jnp.all((spikes >= 0) & (spikes <= 1)))

    def test_lif_neuron(self):
        with brainstate.environ.context(dt=0.1 * u.ms):
            tau = 20.0 * u.ms
            neuron = LIF(self.in_size, tau=tau)
            inputs = self.generate_input()

            # Test initialization
            self.assertEqual(neuron.in_size, (self.in_size,))
            self.assertEqual(neuron.out_size, (self.in_size,))
            self.assertEqual(neuron.tau, tau)

            # Test forward pass
            state = neuron.init_state(self.batch_size)
            call = brainstate.compile.jit(neuron)

            for t in range(self.time_steps):
                out = call(inputs[t])
                self.assertEqual(out.shape, (self.batch_size, self.in_size))

    def test_alif_neuron(self):
        tau = 20.0 * u.ms
        tau_ada = 100.0 * u.ms
        neuron = ALIF(self.in_size, tau=tau, tau_a=tau_ada)
        inputs = self.generate_input()

        # Test initialization
        self.assertEqual(neuron.in_size, (self.in_size,))
        self.assertEqual(neuron.out_size, (self.in_size,))
        self.assertEqual(neuron.tau, tau)
        self.assertEqual(neuron.tau_a, tau_ada)

        # Test forward pass
        neuron.init_state(self.batch_size)
        call = brainstate.compile.jit(neuron)
        with brainstate.environ.context(dt=0.1 * u.ms):
            for t in range(self.time_steps):
                out = call(inputs[t])
                self.assertEqual(out.shape, (self.batch_size, self.in_size))

    def test_spike_function(self):
        for NeuronClass in [IF, LIF, ALIF]:
            neuron = NeuronClass(self.in_size)
            neuron.init_state()
            v = jnp.linspace(-1, 1, self.in_size) * u.mV
            spikes = neuron.get_spike(v)
            self.assertTrue(jnp.all((spikes >= 0) & (spikes <= 1)))

    def test_soft_reset(self):
        for NeuronClass in [IF, LIF, ALIF]:
            neuron = NeuronClass(self.in_size, spk_reset='soft')
            inputs = self.generate_input()
            state = neuron.init_state(self.batch_size)
            call = brainstate.compile.jit(neuron)
            with brainstate.environ.context(dt=0.1 * u.ms):
                for t in range(self.time_steps):
                    out = call(inputs[t])
                    self.assertTrue(jnp.all(neuron.V.value <= neuron.V_th))

    def test_hard_reset(self):
        for NeuronClass in [IF, LIF, ALIF]:
            neuron = NeuronClass(self.in_size, spk_reset='hard')
            inputs = self.generate_input()
            state = neuron.init_state(self.batch_size)
            call = brainstate.compile.jit(neuron)
            with brainstate.environ.context(dt=0.1 * u.ms):
                for t in range(self.time_steps):
                    out = call(inputs[t])
                    self.assertTrue(jnp.all((neuron.V.value < neuron.V_th) | (neuron.V.value == 0. * u.mV)))

    def test_detach_spike(self):
        for NeuronClass in [IF, LIF, ALIF]:
            neuron = NeuronClass(self.in_size)
            inputs = self.generate_input()
            state = neuron.init_state(self.batch_size)
            call = brainstate.compile.jit(neuron)
            with brainstate.environ.context(dt=0.1 * u.ms):
                for t in range(self.time_steps):
                    out = call(inputs[t])
                    self.assertFalse(jax.tree_util.tree_leaves(out)[0].aval.weak_type)

    def test_keep_size(self):
        in_size = (2, 3)
        for NeuronClass in [IF, LIF, ALIF]:
            neuron = NeuronClass(in_size)
            self.assertEqual(neuron.in_size, in_size)
            self.assertEqual(neuron.out_size, in_size)

            inputs = brainstate.random.randn(self.time_steps, self.batch_size, *in_size) * u.mA
            state = neuron.init_state(self.batch_size)
            call = brainstate.compile.jit(neuron)
            with brainstate.environ.context(dt=0.1 * u.ms):
                for t in range(self.time_steps):
                    out = call(inputs[t])
                    self.assertEqual(out.shape, (self.batch_size, *in_size))


if __name__ == '__main__':
    with brainstate.environ.context(dt=0.1):
        unittest.main()
