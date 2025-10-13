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
import braintools
import brainunit as u
import jax
import jax.numpy as jnp

from brainpy.state import HH, MorrisLecar, WangBuzsakiHH


class TestHHNeuron(unittest.TestCase):
    def setUp(self):
        self.in_size = 10
        self.batch_size = 5
        self.time_steps = 100
        self.dt = 0.01 * u.ms

    def generate_input(self):
        return brainstate.random.randn(self.time_steps, self.batch_size, self.in_size) * u.uA

    def test_hh_neuron(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = HH(self.in_size)
            inputs = self.generate_input()

            # Test initialization
            self.assertEqual(neuron.in_size, (self.in_size,))
            self.assertEqual(neuron.out_size, (self.in_size,))

            # Test forward pass
            neuron.init_state(self.batch_size)
            call = brainstate.compile.jit(neuron)

            for t in range(self.time_steps):
                out = call(inputs[t])
                self.assertEqual(out.shape, (self.batch_size, self.in_size))

            # Check state variables
            self.assertEqual(neuron.V.value.shape, (self.batch_size, self.in_size))
            self.assertEqual(neuron.m.value.shape, (self.batch_size, self.in_size))
            self.assertEqual(neuron.h.value.shape, (self.batch_size, self.in_size))
            self.assertEqual(neuron.n.value.shape, (self.batch_size, self.in_size))

    def test_morris_lecar_neuron(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = MorrisLecar(self.in_size)
            inputs = self.generate_input()

            # Test initialization
            self.assertEqual(neuron.in_size, (self.in_size,))
            self.assertEqual(neuron.out_size, (self.in_size,))

            # Test forward pass
            neuron.init_state(self.batch_size)
            call = brainstate.compile.jit(neuron)

            for t in range(self.time_steps):
                out = call(inputs[t])
                self.assertEqual(out.shape, (self.batch_size, self.in_size))

            # Check state variables
            self.assertEqual(neuron.V.value.shape, (self.batch_size, self.in_size))
            self.assertEqual(neuron.W.value.shape, (self.batch_size, self.in_size))

    def test_wang_buzsaki_hh_neuron(self):
        with brainstate.environ.context(dt=self.dt):
            neuron = WangBuzsakiHH(self.in_size)
            inputs = self.generate_input()

            # Test initialization
            self.assertEqual(neuron.in_size, (self.in_size,))
            self.assertEqual(neuron.out_size, (self.in_size,))

            # Test forward pass
            neuron.init_state(self.batch_size)
            call = brainstate.compile.jit(neuron)

            for t in range(self.time_steps):
                out = call(inputs[t])
                self.assertEqual(out.shape, (self.batch_size, self.in_size))

            # Check state variables
            self.assertEqual(neuron.V.value.shape, (self.batch_size, self.in_size))
            self.assertEqual(neuron.h.value.shape, (self.batch_size, self.in_size))
            self.assertEqual(neuron.n.value.shape, (self.batch_size, self.in_size))

    def test_spike_function(self):
        for NeuronClass in [HH, MorrisLecar, WangBuzsakiHH]:
            neuron = NeuronClass(self.in_size)
            neuron.init_state()
            v = jnp.linspace(-80, 40, self.in_size) * u.mV
            spikes = neuron.get_spike(v)
            self.assertTrue(jnp.all((spikes >= 0) & (spikes <= 1)))

    def test_soft_reset(self):
        for NeuronClass in [HH, MorrisLecar, WangBuzsakiHH]:
            neuron = NeuronClass(self.in_size, spk_reset='soft')
            inputs = self.generate_input()
            neuron.init_state(self.batch_size)
            call = brainstate.compile.jit(neuron)
            with brainstate.environ.context(dt=self.dt):
                for t in range(self.time_steps):
                    out = call(inputs[t])
                    # Check that voltage doesn't exceed threshold too much
                    self.assertTrue(jnp.all(neuron.V.value <= neuron.V_th + 20 * u.mV))

    def test_hard_reset(self):
        for NeuronClass in [HH, MorrisLecar, WangBuzsakiHH]:
            neuron = NeuronClass(self.in_size, spk_reset='hard')
            inputs = self.generate_input()
            neuron.init_state(self.batch_size)
            call = brainstate.compile.jit(neuron)
            with brainstate.environ.context(dt=self.dt):
                for t in range(self.time_steps):
                    out = call(inputs[t])
                    # Just check that it runs without error
                    self.assertEqual(out.shape, (self.batch_size, self.in_size))

    def test_detach_spike(self):
        for NeuronClass in [HH, MorrisLecar, WangBuzsakiHH]:
            neuron = NeuronClass(self.in_size)
            inputs = self.generate_input()
            neuron.init_state(self.batch_size)
            call = brainstate.compile.jit(neuron)
            with brainstate.environ.context(dt=self.dt):
                for t in range(self.time_steps):
                    out = call(inputs[t])
                    self.assertFalse(jax.tree_util.tree_leaves(out)[0].aval.weak_type)

    def test_keep_size(self):
        in_size = (2, 3)
        for NeuronClass in [HH, MorrisLecar, WangBuzsakiHH]:
            neuron = NeuronClass(in_size)
            self.assertEqual(neuron.in_size, in_size)
            self.assertEqual(neuron.out_size, in_size)

            inputs = brainstate.random.randn(self.time_steps, self.batch_size, *in_size) * u.uA
            neuron.init_state(self.batch_size)
            call = brainstate.compile.jit(neuron)
            with brainstate.environ.context(dt=self.dt):
                for t in range(self.time_steps):
                    out = call(inputs[t])
                    self.assertEqual(out.shape, (self.batch_size, *in_size))

    def test_hh_gating_variables(self):
        # Test that gating variables are properly initialized and updated
        neuron = HH(self.in_size)
        neuron.init_state(self.batch_size)

        # Check initial values are in valid range [0, 1]
        self.assertTrue(jnp.all((neuron.m.value >= 0) & (neuron.m.value <= 1)))
        self.assertTrue(jnp.all((neuron.h.value >= 0) & (neuron.h.value <= 1)))
        self.assertTrue(jnp.all((neuron.n.value >= 0) & (neuron.n.value <= 1)))

        # Run for some time steps
        inputs = self.generate_input()
        call = brainstate.compile.jit(neuron)
        with brainstate.environ.context(dt=self.dt):
            for t in range(20):
                out = call(inputs[t])

        # Gating variables should still be in valid range
        self.assertTrue(jnp.all((neuron.m.value >= 0) & (neuron.m.value <= 1)))
        self.assertTrue(jnp.all((neuron.h.value >= 0) & (neuron.h.value <= 1)))
        self.assertTrue(jnp.all((neuron.n.value >= 0) & (neuron.n.value <= 1)))

    def test_hh_alpha_beta_functions(self):
        # Test that alpha and beta functions return positive values
        neuron = HH(self.in_size)
        neuron.init_state()

        V_test = jnp.linspace(-80, 40, self.in_size) * u.mV

        m_alpha = neuron.m_alpha(V_test)
        m_beta = neuron.m_beta(V_test)
        h_alpha = neuron.h_alpha(V_test)
        h_beta = neuron.h_beta(V_test)
        n_alpha = neuron.n_alpha(V_test)
        n_beta = neuron.n_beta(V_test)

        # All rate constants should be positive
        if hasattr(m_alpha, 'mantissa'):
            self.assertTrue(jnp.all(m_alpha.mantissa > 0))
            self.assertTrue(jnp.all(m_beta.mantissa > 0))
            self.assertTrue(jnp.all(h_alpha.mantissa > 0))
            self.assertTrue(jnp.all(h_beta.mantissa > 0))
            self.assertTrue(jnp.all(n_alpha.mantissa > 0))
            self.assertTrue(jnp.all(n_beta.mantissa > 0))
        else:
            self.assertTrue(jnp.all(m_alpha > 0))
            self.assertTrue(jnp.all(m_beta > 0))
            self.assertTrue(jnp.all(h_alpha > 0))
            self.assertTrue(jnp.all(h_beta > 0))
            self.assertTrue(jnp.all(n_alpha > 0))
            self.assertTrue(jnp.all(n_beta > 0))

    def test_morris_lecar_steady_states(self):
        # Test that steady-state functions return values in valid range
        neuron = MorrisLecar(self.in_size)
        neuron.init_state()

        V_test = jnp.linspace(-100, 50, self.in_size) * u.mV

        # Manually compute steady states
        M_inf = 0.5 * (1. + u.math.tanh((V_test - neuron.V1) / neuron.V2))
        W_inf = 0.5 * (1. + u.math.tanh((V_test - neuron.V3) / neuron.V4))

        # Steady states should be in [0, 1]
        if hasattr(M_inf, 'mantissa'):
            self.assertTrue(jnp.all((M_inf.mantissa >= 0) & (M_inf.mantissa <= 1)))
            self.assertTrue(jnp.all((W_inf.mantissa >= 0) & (W_inf.mantissa <= 1)))
        else:
            self.assertTrue(jnp.all((M_inf >= 0) & (M_inf <= 1)))
            self.assertTrue(jnp.all((W_inf >= 0) & (W_inf <= 1)))

    def test_wang_buzsaki_m_inf(self):
        # Test that m_inf is properly computed and in valid range
        neuron = WangBuzsakiHH(self.in_size)
        neuron.init_state()

        V_test = jnp.linspace(-80, 40, self.in_size) * u.mV
        m_inf = neuron.m_inf(V_test)

        # m_inf should be in [0, 1]
        if hasattr(m_inf, 'mantissa'):
            self.assertTrue(jnp.all((m_inf.mantissa >= 0) & (m_inf.mantissa <= 1)))
        else:
            self.assertTrue(jnp.all((m_inf >= 0) & (m_inf <= 1)))

    def test_different_parameters(self):
        # Test HH with different conductance values
        hh_custom = HH(
            self.in_size,
            ENa=50. * u.mV,
            gNa=100. * u.msiemens,
            EK=-80. * u.mV,
            gK=30. * u.msiemens
        )
        hh_custom.init_state(self.batch_size)
        self.assertEqual(hh_custom.ENa, 50. * u.mV)
        self.assertEqual(hh_custom.gNa, 100. * u.msiemens)

        # Test MorrisLecar with different parameters
        ml_custom = MorrisLecar(
            self.in_size,
            V_Ca=120. * u.mV,
            g_Ca=4.0 * u.msiemens,
            phi=0.05 / u.ms
        )
        ml_custom.init_state(self.batch_size)
        self.assertEqual(ml_custom.V_Ca, 120. * u.mV)
        self.assertEqual(ml_custom.phi, 0.05 / u.ms)

        # Test WangBuzsakiHH with different phi
        wb_custom = WangBuzsakiHH(
            self.in_size,
            phi=10.0
        )
        wb_custom.init_state(self.batch_size)
        if hasattr(wb_custom.phi, 'mantissa'):
            self.assertEqual(float(wb_custom.phi.mantissa), 10.0)
        else:
            self.assertEqual(float(wb_custom.phi), 10.0)

    def test_ionic_currents(self):
        # Test that ionic currents are computed
        neuron = HH(self.in_size)
        neuron.init_state(self.batch_size)

        # Run one update
        inputs = jnp.ones((self.batch_size, self.in_size)) * 10. * u.uA
        with brainstate.environ.context(dt=self.dt):
            out = neuron.update(inputs)

        # Check that state variables have changed (indicating currents were applied)
        initial_V = braintools.init.param(neuron.V_initializer, neuron.varshape, self.batch_size)
        if hasattr(initial_V, 'mantissa'):
            self.assertFalse(jnp.allclose(neuron.V.value.mantissa, initial_V.mantissa))
        else:
            self.assertFalse(jnp.allclose(neuron.V.value, initial_V))


if __name__ == '__main__':
    unittest.main()
