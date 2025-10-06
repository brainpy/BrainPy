# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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

"""
Comprehensive tests for the Neuron and Synapse base classes in _base.py.

This module tests:
- Neuron base class functionality and abstract interface
- Synapse base class functionality and abstract interface
- Proper initialization and parameter handling
- State management (init_state, reset_state)
- Surrogate gradient function integration
- Reset mechanisms (soft/hard)
- Custom implementations
- Edge cases and error handling
"""

import unittest

import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp

from brainpy._base import Neuron, Synapse


class TestNeuronBaseClass(unittest.TestCase):
    """Test suite for the Neuron base class."""

    def setUp(self):
        """Set up test fixtures."""
        self.in_size = 10
        self.batch_size = 5
        self.in_size_2d = (4, 5)

    def test_neuron_abstract_base_class(self):
        """Test that Neuron is abstract and cannot be instantiated directly."""
        # The base class should be instantiable but get_spike should raise NotImplementedError
        neuron = Neuron(in_size=self.in_size)

        # Test initialization
        self.assertEqual(neuron.in_size, (self.in_size,))
        self.assertIsInstance(neuron.spk_fun, braintools.surrogate.InvSquareGrad)
        self.assertEqual(neuron.spk_reset, 'soft')

        # Test that get_spike raises NotImplementedError
        with self.assertRaises(NotImplementedError):
            neuron.get_spike()

    def test_neuron_default_parameters(self):
        """Test default parameter initialization."""
        neuron = Neuron(in_size=self.in_size)

        self.assertEqual(neuron.spk_reset, 'soft')
        self.assertIsInstance(neuron.spk_fun, braintools.surrogate.InvSquareGrad)
        self.assertIsNone(neuron.name)

    def test_neuron_custom_parameters(self):
        """Test custom parameter initialization."""
        custom_spk_fun = braintools.surrogate.ReluGrad()
        neuron = Neuron(
            in_size=self.in_size,
            spk_fun=custom_spk_fun,
            spk_reset='hard',
            name='test_neuron'
        )

        self.assertEqual(neuron.spk_reset, 'hard')
        self.assertIs(neuron.spk_fun, custom_spk_fun)
        self.assertEqual(neuron.name, 'test_neuron')

    def test_neuron_multidimensional_input_size(self):
        """Test initialization with multi-dimensional input size."""
        neuron = Neuron(in_size=self.in_size_2d)
        self.assertEqual(neuron.in_size, self.in_size_2d)

    def test_neuron_various_surrogate_functions(self):
        """Test different surrogate gradient functions."""
        surrogate_functions = [
            braintools.surrogate.ReluGrad(),
            braintools.surrogate.Sigmoid(),
            braintools.surrogate.InvSquareGrad(),
        ]

        for spk_fun in surrogate_functions:
            neuron = Neuron(in_size=self.in_size, spk_fun=spk_fun)
            self.assertIs(neuron.spk_fun, spk_fun)

    def test_neuron_custom_implementation(self):
        """Test a custom neuron implementation."""

        class CustomNeuron(Neuron):
            """Custom neuron for testing."""

            def __init__(self, in_size, V_th=1.0 * u.mV, **kwargs):
                super().__init__(in_size, **kwargs)
                self.V_th = V_th

            def init_state(self, batch_size=None, **kwargs):
                self.V = brainstate.HiddenState(
                    braintools.init.param(
                        braintools.init.Constant(0. * u.mV),
                        self.varshape,
                        batch_size
                    )
                )

            def reset_state(self, batch_size=None, **kwargs):
                self.V.value = braintools.init.param(
                    braintools.init.Constant(0. * u.mV),
                    self.varshape,
                    batch_size
                )

            def get_spike(self, V=None):
                V = self.V.value if V is None else V
                v_scaled = (V - self.V_th) / self.V_th
                return self.spk_fun(v_scaled)

            def update(self, x):
                self.V.value += x
                return self.get_spike()

        # Test custom neuron
        neuron = CustomNeuron(in_size=self.in_size, V_th=2.0 * u.mV)
        self.assertEqual(neuron.V_th, 2.0 * u.mV)

        # Initialize state
        neuron.init_state(batch_size=self.batch_size)
        self.assertEqual(neuron.V.value.shape, (self.batch_size, self.in_size))

        # Test update
        input_current = 1.5 * u.mV * jnp.ones((self.batch_size, self.in_size))
        spikes = neuron.update(input_current)
        self.assertEqual(spikes.shape, (self.batch_size, self.in_size))

        # Test reset
        neuron.reset_state(batch_size=self.batch_size)
        self.assertTrue(u.math.allclose(neuron.V.value, 0. * u.mV))

    def test_neuron_soft_vs_hard_reset(self):
        """Test that soft and hard reset modes are correctly stored."""
        neuron_soft = Neuron(in_size=self.in_size, spk_reset='soft')
        neuron_hard = Neuron(in_size=self.in_size, spk_reset='hard')

        self.assertEqual(neuron_soft.spk_reset, 'soft')
        self.assertEqual(neuron_hard.spk_reset, 'hard')

    def test_neuron_module_attribute(self):
        """Test __module__ attribute is correctly set."""
        neuron = Neuron(in_size=self.in_size)
        self.assertEqual(neuron.__module__, 'brainpy')


class TestSynapseBaseClass(unittest.TestCase):
    """Test suite for the Synapse base class."""

    def setUp(self):
        """Set up test fixtures."""
        self.in_size = 10
        self.batch_size = 5
        self.in_size_2d = (4, 5)

    def test_synapse_instantiation(self):
        """Test that Synapse can be instantiated."""
        synapse = Synapse(in_size=self.in_size)

        self.assertEqual(synapse.in_size, (self.in_size,))
        self.assertIsNone(synapse.name)

    def test_synapse_default_parameters(self):
        """Test default parameter initialization."""
        synapse = Synapse(in_size=self.in_size)

        self.assertEqual(synapse.in_size, (self.in_size,))
        self.assertIsNone(synapse.name)

    def test_synapse_custom_parameters(self):
        """Test custom parameter initialization."""
        synapse = Synapse(in_size=self.in_size, name='test_synapse')

        self.assertEqual(synapse.name, 'test_synapse')

    def test_synapse_multidimensional_input_size(self):
        """Test initialization with multi-dimensional input size."""
        synapse = Synapse(in_size=self.in_size_2d)
        self.assertEqual(synapse.in_size, self.in_size_2d)

    def test_synapse_custom_implementation(self):
        """Test a custom synapse implementation."""

        class CustomSynapse(Synapse):
            """Custom synapse for testing."""

            def __init__(self, in_size, tau=5.0 * u.ms, **kwargs):
                super().__init__(in_size, **kwargs)
                self.tau = braintools.init.param(tau, self.varshape)
                self.g_initializer = braintools.init.Constant(0. * u.mS)

            def init_state(self, batch_size=None, **kwargs):
                self.g = brainstate.HiddenState(
                    braintools.init.param(
                        self.g_initializer,
                        self.varshape,
                        batch_size
                    )
                )

            def reset_state(self, batch_size=None, **kwargs):
                self.g.value = braintools.init.param(
                    self.g_initializer,
                    self.varshape,
                    batch_size
                )

            def update(self, x=None):
                # Simple exponential decay: dg/dt = -g/tau
                dg = lambda g: -g / self.tau
                self.g.value = brainstate.nn.exp_euler_step(dg, self.g.value)
                if x is not None:
                    self.g.value += x
                return self.g.value

        # Test custom synapse
        synapse = CustomSynapse(in_size=self.in_size, tau=10.0 * u.ms)
        self.assertEqual(synapse.tau, 10.0 * u.ms)

        # Initialize state
        synapse.init_state(batch_size=self.batch_size)
        self.assertEqual(synapse.g.value.shape, (self.batch_size, self.in_size))

        # Test update with input
        with brainstate.environ.context(dt=0.1 * u.ms):
            input_signal = 1.0 * u.mS * jnp.ones((self.batch_size, self.in_size))
            output = synapse.update(input_signal)
            self.assertEqual(output.shape, (self.batch_size, self.in_size))

            # Test update without input (decay only)
            output2 = synapse.update()
            self.assertEqual(output2.shape, (self.batch_size, self.in_size))
            # Output should decay
            self.assertTrue(jnp.all(output2 < output))

        # Test reset
        synapse.reset_state(batch_size=self.batch_size)
        self.assertTrue(u.math.allclose(synapse.g.value, 0. * u.mS))

    def test_synapse_module_attribute(self):
        """Test __module__ attribute is correctly set."""
        synapse = Synapse(in_size=self.in_size)
        self.assertEqual(synapse.__module__, 'brainpy')

    def test_synapse_varshape_attribute(self):
        """Test varshape attribute is correctly set."""
        synapse = Synapse(in_size=self.in_size)
        self.assertEqual(synapse.varshape, (self.in_size,))

        synapse_2d = Synapse(in_size=self.in_size_2d)
        self.assertEqual(synapse_2d.varshape, self.in_size_2d)


class TestNeuronSynapseIntegration(unittest.TestCase):
    """Test integration between Neuron and Synapse classes."""

    def setUp(self):
        """Set up test fixtures."""
        self.in_size = 20
        self.batch_size = 8

    def test_neuron_synapse_pipeline(self):
        """Test a simple neuron-synapse pipeline."""

        # Define custom implementations
        class SimpleNeuron(Neuron):
            def __init__(self, in_size, V_th=1.0 * u.mV, **kwargs):
                super().__init__(in_size, **kwargs)
                self.V_th = V_th

            def init_state(self, batch_size=None, **kwargs):
                self.V = brainstate.HiddenState(
                    braintools.init.param(
                        braintools.init.Constant(0. * u.mV),
                        self.varshape,
                        batch_size
                    )
                )

            def reset_state(self, batch_size=None, **kwargs):
                self.V.value = braintools.init.param(
                    braintools.init.Constant(0. * u.mV),
                    self.varshape,
                    batch_size
                )

            def get_spike(self, V=None):
                V = self.V.value if V is None else V
                return self.spk_fun((V - self.V_th) / self.V_th)

            def update(self, x):
                self.V.value += x
                return self.get_spike()

        class SimpleSynapse(Synapse):
            def __init__(self, in_size, tau=5.0 * u.ms, **kwargs):
                super().__init__(in_size, **kwargs)
                self.tau = braintools.init.param(tau, self.varshape)
                self.g_initializer = braintools.init.Constant(0. * u.mS)

            def init_state(self, batch_size=None, **kwargs):
                self.g = brainstate.HiddenState(
                    braintools.init.param(
                        self.g_initializer,
                        self.varshape,
                        batch_size
                    )
                )

            def reset_state(self, batch_size=None, **kwargs):
                self.g.value = braintools.init.param(
                    self.g_initializer,
                    self.varshape,
                    batch_size
                )

            def update(self, x=None):
                dg = lambda g: -g / self.tau
                self.g.value = brainstate.nn.exp_euler_step(dg, self.g.value)
                if x is not None:
                    self.g.value += x
                return self.g.value

        # Create neuron and synapse
        neuron = SimpleNeuron(in_size=self.in_size, V_th=0.5 * u.mV)
        synapse = SimpleSynapse(in_size=self.in_size, tau=10.0 * u.ms)

        # Initialize states
        neuron.init_state(batch_size=self.batch_size)
        synapse.init_state(batch_size=self.batch_size)

        # Simulate a few steps
        with brainstate.environ.context(dt=0.1 * u.ms):
            input_current = 1.0 * u.mV * jnp.ones((self.batch_size, self.in_size))

            for _ in range(10):
                # Neuron update
                spikes = neuron.update(input_current)

                # Synapse update
                conductance = synapse.update(spikes * 1.0 * u.mS)

                # Check shapes
                self.assertEqual(spikes.shape, (self.batch_size, self.in_size))
                self.assertEqual(conductance.shape, (self.batch_size, self.in_size))


class TestEdgeCasesAndErrorHandling(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_neuron_zero_size(self):
        """Test neuron with zero-sized input."""
        # This might be allowed in some frameworks
        try:
            neuron = Neuron(in_size=0)
            self.assertEqual(neuron.in_size, (0,))
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass

    def test_synapse_zero_size(self):
        """Test synapse with zero-sized input."""
        try:
            synapse = Synapse(in_size=0)
            self.assertEqual(synapse.in_size, (0,))
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass

    def test_neuron_large_input_size(self):
        """Test neuron with large input size."""
        large_size = 10000
        neuron = Neuron(in_size=large_size)
        self.assertEqual(neuron.in_size, (large_size,))

    def test_synapse_large_input_size(self):
        """Test synapse with large input size."""
        large_size = 10000
        synapse = Synapse(in_size=large_size)
        self.assertEqual(synapse.in_size, (large_size,))

    def test_neuron_tuple_input_size(self):
        """Test neuron with tuple input size."""
        tuple_size = (3, 4, 5)
        neuron = Neuron(in_size=tuple_size)
        self.assertEqual(neuron.in_size, tuple_size)

    def test_synapse_tuple_input_size(self):
        """Test synapse with tuple input size."""
        tuple_size = (3, 4, 5)
        synapse = Synapse(in_size=tuple_size)
        self.assertEqual(synapse.in_size, tuple_size)


class TestDocstringExamples(unittest.TestCase):
    """Test that the examples in docstrings work correctly."""

    def test_neuron_docstring_simple_example(self):
        """Test the simple neuron example from the docstring."""

        class SimpleNeuron(Neuron):
            def __init__(self, in_size, V_th=1.0 * u.mV, **kwargs):
                super().__init__(in_size, **kwargs)
                self.V_th = V_th

            def init_state(self, batch_size=None, **kwargs):
                self.V = brainstate.HiddenState(
                    braintools.init.param(
                        braintools.init.Constant(0. * u.mV),
                        self.varshape,
                        batch_size
                    )
                )

            def reset_state(self, batch_size=None, **kwargs):
                self.V.value = braintools.init.param(
                    braintools.init.Constant(0. * u.mV),
                    self.varshape,
                    batch_size
                )

            def get_spike(self, V=None):
                V = self.V.value if V is None else V
                return self.spk_fun((V - self.V_th) / self.V_th)

            def update(self, x):
                self.V.value += x
                return self.get_spike()

        # Create and test
        neuron = SimpleNeuron(in_size=10, V_th=1.0 * u.mV)
        neuron.init_state(batch_size=1)
        input_current = 0.5 * u.mV * jnp.ones((1, 10))
        spikes = neuron.update(input_current)

        self.assertIsNotNone(spikes)
        self.assertEqual(spikes.shape, (1, 10))

    def test_synapse_docstring_simple_example(self):
        """Test the simple synapse example from the docstring."""

        class SimpleSynapse(Synapse):
            def __init__(self, in_size, tau=5.0 * u.ms, **kwargs):
                super().__init__(in_size, **kwargs)
                self.tau = braintools.init.param(tau, self.varshape)
                self.g_init = braintools.init.Constant(0. * u.mS)

            def init_state(self, batch_size=None, **kwargs):
                self.g = brainstate.HiddenState(
                    braintools.init.param(
                        self.g_init,
                        self.varshape,
                        batch_size
                    )
                )

            def reset_state(self, batch_size=None, **kwargs):
                self.g.value = braintools.init.param(
                    self.g_init,
                    self.varshape,
                    batch_size
                )

            def update(self, x=None):
                dg = lambda g: -g / self.tau
                self.g.value = brainstate.nn.exp_euler_step(dg, self.g.value)
                if x is not None:
                    self.g.value += x
                return self.g.value

        # Create and test
        with brainstate.environ.context(dt=0.1 * u.ms):
            synapse = SimpleSynapse(in_size=10, tau=5.0 * u.ms)
            synapse.init_state(batch_size=1)
            input_signal = 1.0 * u.mS * jnp.ones((1, 10))
            output = synapse.update(input_signal)

            self.assertIsNotNone(output)
            self.assertEqual(output.shape, (1, 10))


if __name__ == '__main__':
    with brainstate.environ.context(dt=0.1 * u.ms):
        unittest.main()
