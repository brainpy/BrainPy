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
        brainstate.environ.set(dt=0.1 * u.ms)
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
            call = brainstate.transform.jit(neuron)

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
            call = brainstate.transform.jit(neuron)

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
            call = brainstate.transform.jit(neuron)

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

    def test_no_repeat_spike_while_above_threshold(self):
        """B4 regression: HH-family neurons do not reset V, so the membrane stays
        above threshold for the whole action potential (~1-2 ms). A per-step
        ``get_spike()`` therefore reported one spike on every step above threshold.
        A neuron that starts a step already above threshold has no rising edge and
        must emit no spike."""
        for NeuronClass in [HH, MorrisLecar, WangBuzsakiHH]:
            neuron = NeuronClass(1)
            neuron.init_state()
            # Force the membrane well above threshold (as during an AP plateau).
            neuron.V.value = jnp.full(neuron.V.value.shape, 100.) * u.mV
            with brainstate.environ.context(dt=self.dt):
                spike = neuron.update()
            self.assertAlmostEqual(
                float(u.get_magnitude(spike).ravel()[0]), 0.0, places=6,
                msg=f'{NeuronClass.__name__} re-emitted a spike while already above threshold',
            )

    def test_single_spike_per_action_potential(self):
        """B4 regression: under constant drive the reported spike count must match
        the number of threshold crossings (action potentials), not the number of
        steps spent above threshold."""
        with brainstate.environ.context(dt=0.01 * u.ms):
            # Deterministic start at rest: the default V_initializer draws V0 from
            # Uniform(-70, -60) mV via the unseeded global RNG, and for V0 close to
            # -60 mV the first action potential under 10 uA arrives after the 25 ms
            # window, so the test passed or failed depending on test ordering.
            neuron = HH(1, V_initializer=braintools.init.Constant(-65. * u.mV))
            neuron.init_state()

            def step(i):
                s = neuron.update(x=10. * u.uA)
                return u.get_magnitude(s).ravel()[0], u.get_magnitude(neuron.V.value).ravel()[0]

            spikes, Vs = brainstate.transform.for_loop(step, jnp.arange(2500))

        spikes = jnp.asarray(spikes)
        Vs = jnp.asarray(Vs)
        V_th = float(jnp.asarray(u.get_magnitude(neuron.V_th)).ravel()[0])
        spike_steps = int((spikes > 0.5).sum())
        above_steps = int((Vs > V_th).sum())
        rising_edges = int(((Vs[1:] > V_th) & (Vs[:-1] <= V_th)).sum())

        self.assertGreaterEqual(rising_edges, 1)        # it actually spiked
        self.assertLess(spike_steps, above_steps)       # not one spike per above-step
        self.assertEqual(spike_steps, rising_edges)     # exactly one spike per crossing

    def test_soft_reset(self):
        for NeuronClass in [HH, MorrisLecar, WangBuzsakiHH]:
            neuron = NeuronClass(self.in_size, spk_reset='soft')
            inputs = self.generate_input()
            neuron.init_state(self.batch_size)
            call = brainstate.transform.jit(neuron)
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
            call = brainstate.transform.jit(neuron)
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
            call = brainstate.transform.jit(neuron)
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
            call = brainstate.transform.jit(neuron)
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
        call = brainstate.transform.jit(neuron)
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

    def test_wang_buzsaki_n_alpha_rate(self):
        r"""Regression: the K+ activation rate must match Wang & Buzsaki (1996),
        ``alpha_n = 0.01 (V + 34) / (1 - exp(-0.1 (V + 34)))`` per ms, not 10x it.
        """
        neuron = WangBuzsakiHH(self.in_size)
        neuron.init_state()
        # Avoid the removable singularity at V = -34 mV.
        V_test = jnp.linspace(-70., -40., self.in_size) * u.mV
        v = u.get_magnitude(V_test / u.mV)
        canonical = 0.01 * (v + 34.) / (1. - jnp.exp(-0.1 * (v + 34.))) / u.ms
        ratio = u.get_magnitude(neuron.n_alpha(V_test) / canonical)
        self.assertTrue(jnp.allclose(ratio, 1.0, rtol=1e-4))

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
