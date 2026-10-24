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
import numpy as np
import numpy.testing as npt
from brainpy.state import SpikeTime, PoissonSpike, PoissonEncoder, poisson_input
from brainpy.state import (
    SectionInput, ConstantInput, StepInput, RampInput,
    SinusoidalInput, OUProcessInput, WienerProcessInput,
)


# ---------------------------------------------------------------------------
# SpikeTime — Construction
# ---------------------------------------------------------------------------

class TestSpikeTimeConstruction(unittest.TestCase):
    r"""Tests for SpikeTime.__init__ validation and CSR construction."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def test_basic_construction(self):
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(3, times=[10 * u.ms, 20 * u.ms], indices=[0, 1])
            self.assertEqual(st.varshape, (3,))
            self.assertEqual(st.num_times, 2)

    def test_default_weight_is_float(self):
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(2, times=[10 * u.ms], indices=[0])
            self.assertEqual(st._spk_dtype, brainstate.environ.dftype())
            self.assertTrue(st._scalar_weight)

    def test_scalar_weight_custom(self):
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(2, times=[10 * u.ms], indices=[0], weights=2.5)
            self.assertTrue(st._scalar_weight)

    def test_per_event_weights(self):
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(2, times=[10 * u.ms, 20 * u.ms], indices=[0, 1],
                           weights=[0.5, 0.8])
            self.assertFalse(st._scalar_weight)

    def test_mismatched_lengths_raises(self):
        with brainstate.environ.context(dt=self.dt):
            with self.assertRaises(ValueError):
                SpikeTime(2, times=[10 * u.ms, 20 * u.ms], indices=[0])

    def test_invalid_time_as_step_raises(self):
        with brainstate.environ.context(dt=self.dt):
            with self.assertRaises(ValueError):
                SpikeTime(2, times=[10 * u.ms], indices=[0], time_as_step='bad')

    def test_per_event_weights_wrong_length_raises(self):
        with brainstate.environ.context(dt=self.dt):
            with self.assertRaises(ValueError):
                SpikeTime(2, times=[10 * u.ms, 20 * u.ms], indices=[0, 1],
                          weights=[1.0, 2.0, 3.0])

    def test_empty_events(self):
        r"""Empty events with explicit Quantity arrays."""
        with brainstate.environ.context(dt=self.dt):
            ditype = brainstate.environ.ditype()
            st = SpikeTime(3,
                           times=u.math.zeros(0) * u.ms,
                           indices=jnp.zeros(0, dtype=ditype))
            self.assertEqual(st.num_times, 0)
            result = st.update(index=50)
            npt.assert_array_equal(result, jnp.zeros(3, dtype=st._spk_dtype))

    def test_module_attribute(self):
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(2, times=[10 * u.ms], indices=[0])
            self.assertEqual(st.__module__, 'brainpy.state')

    def test_custom_name(self):
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(2, times=[10 * u.ms], indices=[0], name='my_spike')
            self.assertEqual(st.name, 'my_spike')


# ---------------------------------------------------------------------------
# SpikeTime — update() call paths
# ---------------------------------------------------------------------------

class TestSpikeTimeUpdate(unittest.TestCase):
    r"""Tests for the three update() call paths: index, time, environ."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _make(self):
        return SpikeTime(3,
                         times=[10 * u.ms, 20 * u.ms, 30 * u.ms],
                         indices=[0, 1, 2])

    def test_update_with_explicit_index(self):
        with brainstate.environ.context(dt=self.dt):
            st = self._make()
            result = st.update(index=100)  # step 100 = 10 ms
            self.assertGreater(float(result[0]), 0)
            self.assertAlmostEqual(float(result[1]), 0, places=5)
            self.assertAlmostEqual(float(result[2]), 0, places=5)

    def test_update_with_explicit_time(self):
        with brainstate.environ.context(dt=self.dt):
            st = self._make()
            result = st.update(time=20.0 * u.ms)
            self.assertAlmostEqual(float(result[0]), 0, places=5)
            self.assertGreater(float(result[1]), 0)
            self.assertAlmostEqual(float(result[2]), 0, places=5)

    def test_update_environ_with_i(self):
        with brainstate.environ.context(dt=self.dt):
            st = self._make()
            with brainstate.environ.context(i=300):
                result = st.update()
                self.assertAlmostEqual(float(result[0]), 0, places=5)
                self.assertAlmostEqual(float(result[1]), 0, places=5)
                self.assertGreater(float(result[2]), 0)

    def test_update_environ_with_t_only(self):
        with brainstate.environ.context(dt=self.dt):
            st = self._make()
            with brainstate.environ.context(t=10.0 * u.ms):
                result = st.update()
                self.assertGreater(float(result[0]), 0)
                self.assertAlmostEqual(float(result[1]), 0, places=5)

    def test_update_no_spike_step(self):
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(3, times=[10 * u.ms, 20 * u.ms], indices=[0, 1])
            result = st.update(index=50)  # 5 ms — no events
            npt.assert_array_equal(result, jnp.zeros(3, dtype=st._spk_dtype))

    def test_update_out_of_bounds_step(self):
        r"""Steps beyond the max event step should return zeros."""
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(3, times=[10 * u.ms], indices=[0])
            result = st.update(index=99999)
            npt.assert_array_equal(result, jnp.zeros(3, dtype=st._spk_dtype))

    def test_update_step_zero(self):
        r"""Step 0 with no events at time 0 returns zeros."""
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(3, times=[10 * u.ms], indices=[0])
            result = st.update(index=0)
            npt.assert_array_equal(result, jnp.zeros(3, dtype=st._spk_dtype))


# ---------------------------------------------------------------------------
# SpikeTime — Simultaneous events
# ---------------------------------------------------------------------------

class TestSpikeTimeSimultaneous(unittest.TestCase):
    r"""Tests for multiple events at the same time step."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def test_two_neurons_same_step(self):
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(3,
                           times=[10 * u.ms, 10 * u.ms, 20 * u.ms],
                           indices=[0, 1, 2])
            result = st.update(index=100)
            self.assertGreater(float(result[0]), 0)
            self.assertGreater(float(result[1]), 0)
            self.assertAlmostEqual(float(result[2]), 0, places=5)

    def test_many_neurons_same_step(self):
        with brainstate.environ.context(dt=self.dt):
            n = 10
            st = SpikeTime(n,
                           times=[5 * u.ms] * n,
                           indices=list(range(n)))
            result = st.update(index=50)
            for j in range(n):
                self.assertGreater(float(result[j]), 0)


# ---------------------------------------------------------------------------
# SpikeTime — Weights
# ---------------------------------------------------------------------------

class TestSpikeTimeWeights(unittest.TestCase):
    r"""Tests for different weight modes."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def test_default_weight_value(self):
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(2, times=[10 * u.ms], indices=[0])
            result = st.update(index=100)
            self.assertAlmostEqual(float(result[0]), 1.0, places=5)
            self.assertAlmostEqual(float(result[1]), 0.0, places=5)

    def test_scalar_float_weight(self):
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(2, times=[10 * u.ms, 20 * u.ms], indices=[0, 1],
                           weights=2.5)
            result = st.update(index=100)
            self.assertAlmostEqual(float(result[0]), 2.5, places=4)
            self.assertAlmostEqual(float(result[1]), 0.0, places=5)

    def test_per_event_weights_correctness(self):
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(2,
                           times=[10 * u.ms, 20 * u.ms],
                           indices=[0, 1],
                           weights=[0.5, 0.8])
            r1 = st.update(index=100)
            self.assertAlmostEqual(float(r1[0]), 0.5, places=4)
            self.assertAlmostEqual(float(r1[1]), 0.0, places=5)

            r2 = st.update(index=200)
            self.assertAlmostEqual(float(r2[0]), 0.0, places=5)
            self.assertAlmostEqual(float(r2[1]), 0.8, places=4)

    def test_zero_weight(self):
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(2, times=[10 * u.ms], indices=[0], weights=0.0)
            result = st.update(index=100)
            self.assertAlmostEqual(float(result[0]), 0.0, places=5)


# ---------------------------------------------------------------------------
# SpikeTime — Rounding modes
# ---------------------------------------------------------------------------

class TestSpikeTimeRounding(unittest.TestCase):
    r"""Tests for time_as_step rounding modes."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def test_round_mode(self):
        with brainstate.environ.context(dt=self.dt):
            # 10.04 ms / 0.1 ms = 100.4  -> round -> 100
            st = SpikeTime(2, times=[10.04 * u.ms], indices=[0],
                           time_as_step='round')
            result = st.update(index=100)
            self.assertGreater(float(result[0]), 0.0)

    def test_floor_mode(self):
        with brainstate.environ.context(dt=self.dt):
            # 10.09 ms / 0.1 ms = 100.9  -> floor -> 100
            st = SpikeTime(2, times=[10.09 * u.ms], indices=[0],
                           time_as_step='floor')
            result = st.update(index=100)
            self.assertGreater(float(result[0]), 0.0)

    def test_ceil_mode(self):
        with brainstate.environ.context(dt=self.dt):
            # 10.01 ms / 0.1 ms = 100.1  -> ceil -> 101
            st = SpikeTime(2, times=[10.01 * u.ms], indices=[0],
                           time_as_step='ceil')
            result = st.update(index=101)
            self.assertGreater(float(result[0]), 0.0)
            # Should NOT fire at step 100
            result100 = st.update(index=100)
            self.assertAlmostEqual(float(result100[0]), 0.0, places=5)

    def test_rounding_property(self):
        with brainstate.environ.context(dt=self.dt):
            st_floor = SpikeTime(2, times=[10 * u.ms], indices=[0],
                                 time_as_step='floor')
            st_round = SpikeTime(2, times=[10 * u.ms], indices=[0],
                                 time_as_step='round')
            st_ceil = SpikeTime(2, times=[10 * u.ms], indices=[0],
                                time_as_step='ceil')
            self.assertEqual(st_floor.rounding, u.math.floor)
            self.assertEqual(st_round.rounding, u.math.round)
            self.assertEqual(st_ceil.rounding, u.math.ceil)


# ---------------------------------------------------------------------------
# SpikeTime — Sorting
# ---------------------------------------------------------------------------

class TestSpikeTimeSorting(unittest.TestCase):
    r"""Tests that events are always sorted correctly during construction."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def test_unsorted_input_produces_correct_output(self):
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(3,
                           times=[30 * u.ms, 10 * u.ms, 20 * u.ms],
                           indices=[2, 0, 1])
            r1 = st.update(index=100)
            self.assertGreater(float(r1[0]), 0)
            self.assertAlmostEqual(float(r1[2]), 0, places=5)

            r2 = st.update(index=300)
            self.assertGreater(float(r2[2]), 0)
            self.assertAlmostEqual(float(r2[0]), 0, places=5)

    def test_per_event_weights_sorted_correctly(self):
        r"""Weights must follow their original event after sorting."""
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(3,
                           times=[30 * u.ms, 10 * u.ms, 20 * u.ms],
                           indices=[2, 0, 1],
                           weights=[0.3, 0.1, 0.2])
            # event (10ms, neuron 0) should have weight 0.1
            r1 = st.update(index=100)
            self.assertAlmostEqual(float(r1[0]), 0.1, places=4)

            # event (20ms, neuron 1) should have weight 0.2
            r2 = st.update(index=200)
            self.assertAlmostEqual(float(r2[1]), 0.2, places=4)

            # event (30ms, neuron 2) should have weight 0.3
            r3 = st.update(index=300)
            self.assertAlmostEqual(float(r3[2]), 0.3, places=4)


# ---------------------------------------------------------------------------
# SpikeTime — JIT compatibility
# ---------------------------------------------------------------------------

class TestSpikeTimeJIT(unittest.TestCase):
    r"""Tests for JIT compatibility via brainstate.transform.for_loop and jax.jit."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def test_for_loop_scalar_weight(self):
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(3,
                           times=[1 * u.ms, 2 * u.ms, 3 * u.ms],
                           indices=[0, 1, 2])

            def step(t, i):
                with brainstate.environ.context(t=t, i=i):
                    return st.update()

            times = u.math.arange(0. * u.ms, 5. * u.ms, self.dt)
            indices = u.math.arange(times.size)
            spikes = brainstate.transform.for_loop(step, times, indices)

            # step 10 = 1 ms -> neuron 0 fires
            self.assertGreater(float(spikes[10][0]), 0)
            self.assertAlmostEqual(float(spikes[10][1]), 0, places=5)
            # step 20 = 2 ms -> neuron 1 fires
            self.assertGreater(float(spikes[20][1]), 0)
            # step 30 = 3 ms -> neuron 2 fires
            self.assertGreater(float(spikes[30][2]), 0)
            # step 5 = 0.5 ms -> no spikes
            npt.assert_array_equal(spikes[5], jnp.zeros(3, dtype=spikes.dtype))

    def test_for_loop_per_event_weights(self):
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(3,
                           times=[1 * u.ms, 2 * u.ms, 3 * u.ms],
                           indices=[0, 1, 2],
                           weights=[0.5, 0.8, 1.2])

            def step(t, i):
                with brainstate.environ.context(t=t, i=i):
                    return st.update()

            times = u.math.arange(0. * u.ms, 5. * u.ms, self.dt)
            indices = u.math.arange(times.size)
            spikes = brainstate.transform.for_loop(step, times, indices)

            self.assertAlmostEqual(float(spikes[10][0]), 0.5, places=4)
            self.assertAlmostEqual(float(spikes[20][1]), 0.8, places=4)
            self.assertAlmostEqual(float(spikes[30][2]), 1.2, places=4)

    def test_jit_compiled_update_with_index(self):
        r"""Direct jax.jit wrapping of update(index=...)."""
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(3,
                           times=[10 * u.ms, 20 * u.ms],
                           indices=[0, 1])

            @jax.jit
            def run(idx):
                return st.update(index=idx)

            result = run(100)
            self.assertGreater(float(result[0]), 0)
            self.assertAlmostEqual(float(result[1]), 0, places=5)

            result2 = run(200)
            self.assertAlmostEqual(float(result2[0]), 0, places=5)
            self.assertGreater(float(result2[1]), 0)


# ---------------------------------------------------------------------------
# SpikeTime — Dimensionless times
# ---------------------------------------------------------------------------

class TestSpikeTimeDimenslessTimes(unittest.TestCase):
    r"""Tests with dimensionless (plain float) spike times."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def test_float_times_with_units(self):
        r"""Float times with explicit units."""
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(2, times=[10.0 * u.ms, 20.0 * u.ms], indices=[0, 1])
            self.assertEqual(st.num_times, 2)
            result = st.update(index=100)
            self.assertGreater(float(result[0]), 0)
            self.assertAlmostEqual(float(result[1]), 0, places=5)


# ---------------------------------------------------------------------------
# SpikeTime — Large scale
# ---------------------------------------------------------------------------

class TestSpikeTimeLargeScale(unittest.TestCase):
    r"""Test with a larger number of events."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def test_many_events(self):
        n_neurons = 100
        n_events = 500
        with brainstate.environ.context(dt=self.dt):
            brainstate.random.seed(42)
            times_arr = brainstate.random.uniform(1.0, 50.0, (n_events,)) * u.ms
            indices_arr = brainstate.random.randint(0, n_neurons, (n_events,))
            st = SpikeTime(n_neurons, times=times_arr, indices=indices_arr)
            result = st.update(index=100)
            self.assertEqual(result.shape, (n_neurons,))

    def test_single_event(self):
        with brainstate.environ.context(dt=self.dt):
            st = SpikeTime(5, times=[10 * u.ms], indices=[3])
            result = st.update(index=100)
            self.assertGreater(float(result[3]), 0)
            for j in [0, 1, 2, 4]:
                self.assertAlmostEqual(float(result[j]), 0, places=5)


# ---------------------------------------------------------------------------
# PoissonSpike
# ---------------------------------------------------------------------------

class TestPoissonSpike(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms
        self.in_size = 100

    def test_construction(self):
        with brainstate.environ.context(dt=self.dt):
            ps = PoissonSpike(self.in_size, freqs=100 * u.Hz)
            self.assertEqual(ps.varshape, (self.in_size,))
            self.assertEqual(ps.spk_type, bool)
            self.assertEqual(ps.__module__, 'brainpy.state')

    def test_output_shape(self):
        with brainstate.environ.context(dt=self.dt):
            ps = PoissonSpike(self.in_size, freqs=100 * u.Hz)
            result = ps.update()
            self.assertEqual(result.shape, (self.in_size,))

    def test_output_dtype_bool(self):
        with brainstate.environ.context(dt=self.dt):
            ps = PoissonSpike(self.in_size, freqs=100 * u.Hz, spk_type=bool)
            result = ps.update()
            self.assertEqual(result.dtype, jnp.bool_)

    def test_output_dtype_float(self):
        with brainstate.environ.context(dt=self.dt):
            ps = PoissonSpike(self.in_size, freqs=100 * u.Hz, spk_type=float)
            result = ps.update()
            self.assertTrue(jnp.issubdtype(result.dtype, jnp.floating))

    def test_high_frequency_produces_spikes(self):
        r"""With very high frequency and many neurons, at least some should spike."""
        with brainstate.environ.context(dt=self.dt):
            brainstate.random.seed(0)
            ps = PoissonSpike(1000, freqs=5000 * u.Hz)
            result = ps.update()
            self.assertGreater(int(result.sum()), 0)

    def test_zero_frequency_no_spikes(self):
        with brainstate.environ.context(dt=self.dt):
            ps = PoissonSpike(100, freqs=0 * u.Hz)
            result = ps.update()
            npt.assert_array_equal(result, jnp.zeros(100, dtype=bool))

    def test_scalar_frequency(self):
        with brainstate.environ.context(dt=self.dt):
            ps = PoissonSpike(10, freqs=50 * u.Hz)
            result = ps.update()
            self.assertEqual(result.shape, (10,))

    def test_array_frequency(self):
        with brainstate.environ.context(dt=self.dt):
            freqs = jnp.linspace(10, 200, 10) * u.Hz
            ps = PoissonSpike(10, freqs=freqs)
            result = ps.update()
            self.assertEqual(result.shape, (10,))


# ---------------------------------------------------------------------------
# PoissonEncoder
# ---------------------------------------------------------------------------

class TestPoissonEncoder(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms
        self.in_size = 50

    def test_construction(self):
        with brainstate.environ.context(dt=self.dt):
            enc = PoissonEncoder(self.in_size)
            self.assertEqual(enc.varshape, (self.in_size,))
            self.assertEqual(enc.__module__, 'brainpy.state')

    def test_output_shape(self):
        with brainstate.environ.context(dt=self.dt):
            enc = PoissonEncoder(self.in_size)
            rates = jnp.ones(self.in_size) * 100 * u.Hz
            result = enc.update(rates)
            self.assertEqual(result.shape, (self.in_size,))

    def test_output_dtype_bool(self):
        with brainstate.environ.context(dt=self.dt):
            enc = PoissonEncoder(self.in_size, spk_type=bool)
            rates = jnp.ones(self.in_size) * 100 * u.Hz
            result = enc.update(rates)
            self.assertEqual(result.dtype, jnp.bool_)

    def test_output_dtype_float(self):
        with brainstate.environ.context(dt=self.dt):
            enc = PoissonEncoder(self.in_size, spk_type=float)
            rates = jnp.ones(self.in_size) * 100 * u.Hz
            result = enc.update(rates)
            self.assertTrue(jnp.issubdtype(result.dtype, jnp.floating))

    def test_zero_rate_no_spikes(self):
        with brainstate.environ.context(dt=self.dt):
            enc = PoissonEncoder(20)
            rates = jnp.zeros(20) * u.Hz
            result = enc.update(rates)
            npt.assert_array_equal(result, jnp.zeros(20, dtype=bool))

    def test_dynamic_rates(self):
        r"""Encoder accepts different rates each call."""
        with brainstate.environ.context(dt=self.dt):
            enc = PoissonEncoder(10)
            r1 = enc.update(jnp.ones(10) * 0 * u.Hz)
            r2 = enc.update(jnp.ones(10) * 100 * u.Hz)
            npt.assert_array_equal(r1, jnp.zeros(10, dtype=bool))
            self.assertEqual(r2.shape, (10,))


# ---------------------------------------------------------------------------
# poisson_input (function)
# ---------------------------------------------------------------------------

class TestPoissonInputFunction(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def test_updates_target_all_indices(self):
        with brainstate.environ.context(dt=self.dt):
            brainstate.random.seed(42)
            V = brainstate.HiddenState(jnp.zeros(10) * u.mV)
            original = V.value.copy()
            poisson_input(
                freq=1000 * u.Hz,
                num_input=200,
                weight=0.1 * u.mV,
                target=V,
            )
            self.assertFalse(jnp.allclose(
                V.value.to_decimal(u.mV),
                original.to_decimal(u.mV)
            ))

    def test_finite_when_rate_dt_exceeds_one(self):
        r"""Regression: freq*dt > 1 must not yield NaN (per-bin probability is
        bounded by 1 - exp(-freq*dt))."""
        with brainstate.environ.context(dt=1.0 * u.ms):
            V = brainstate.HiddenState(jnp.zeros(50) * u.mV)
            poisson_input(freq=20000. * u.Hz, num_input=100, weight=0.1 * u.mV, target=V)
            self.assertTrue(jnp.all(jnp.isfinite(V.value.to_decimal(u.mV))))

    def test_gaussian_branch_nonnegative_integer_counts(self):
        r"""Regression: in the Gaussian-approximation regime the injected
        counts must be non-negative integers, like the binomial branch."""
        with brainstate.environ.context(dt=self.dt):
            V = brainstate.HiddenState(jnp.zeros(2000) * u.mV)
            poisson_input(freq=5000. * u.Hz, num_input=1000, weight=1.0 * u.mV, target=V)
            counts = V.value.to_decimal(u.mV)  # weight = 1 mV -> magnitude == count
            self.assertTrue(jnp.all(counts >= 0.))
            self.assertTrue(jnp.allclose(counts, jnp.round(counts)))

    def test_updates_target_specific_indices(self):
        with brainstate.environ.context(dt=self.dt):
            brainstate.random.seed(42)
            V = brainstate.HiddenState(jnp.zeros(10) * u.mV)
            idx = np.array([0, 5, 9])
            poisson_input(
                freq=1000 * u.Hz,
                num_input=200,
                weight=0.1 * u.mV,
                target=V,
                indices=idx,
            )
            val = V.value.to_decimal(u.mV)
            # Non-targeted indices should remain zero
            for i in [1, 2, 3, 4, 6, 7, 8]:
                self.assertAlmostEqual(float(val[i]), 0.0, places=5)

    def test_refractory_mask(self):
        with brainstate.environ.context(dt=self.dt):
            brainstate.random.seed(42)
            V = brainstate.HiddenState(jnp.zeros(10) * u.mV)
            refractory = jnp.array([True] * 5 + [False] * 5)
            poisson_input(
                freq=5000 * u.Hz,
                num_input=500,
                weight=0.1 * u.mV,
                target=V,
                refractory=refractory,
            )
            val = V.value.to_decimal(u.mV)
            for i in range(5):
                self.assertAlmostEqual(float(val[i]), 0.0, places=5)

    def test_target_must_be_state(self):
        with brainstate.environ.context(dt=self.dt):
            with self.assertRaises(AssertionError):
                poisson_input(
                    freq=100 * u.Hz,
                    num_input=10,
                    weight=0.1 * u.mV,
                    target=jnp.zeros(5),  # not a State
                )

    def test_no_refractory(self):
        r"""Without refractory, all neurons can be updated."""
        with brainstate.environ.context(dt=self.dt):
            brainstate.random.seed(42)
            V = brainstate.HiddenState(jnp.zeros(10) * u.mV)
            poisson_input(
                freq=5000 * u.Hz,
                num_input=500,
                weight=0.1 * u.mV,
                target=V,
            )
            val = V.value.to_decimal(u.mV)
            # With high freq, at least some should be nonzero
            self.assertGreater(float(jnp.abs(val).sum()), 0.0)


# ---------------------------------------------------------------------------
# Analog current generators (per-step Modules)
# ---------------------------------------------------------------------------

class _AnalogInputTestBase(unittest.TestCase):
    r"""Shared driver: step a generator through its own clock via ``environ['i']``."""

    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.dt = 0.1 * u.ms

    def _run(self, gen, n_steps, unit):
        out = []

        @brainstate.transform.jit
        def _step(i):
            with brainstate.environ.context(i=i):
                return gen.update()

        with brainstate.environ.context(dt=self.dt):
            brainstate.nn.init_all_states(gen)
            for k in range(n_steps):
                out.append(np.asarray(_step(k).to_decimal(unit)))
        return np.stack(out, axis=0)


class TestConstantInput(_AnalogInputTestBase):
    def test_value_within_and_zero_after_window(self):
        gen = ConstantInput(3, 2.0 * u.nA, duration=0.5 * u.ms)  # 5 steps
        out = self._run(gen, 7, u.nA)
        self.assertEqual(out.shape, (7, 3))
        npt.assert_allclose(out[:5], 2.0)
        npt.assert_allclose(out[5:], 0.0)


class TestRampInput(_AnalogInputTestBase):
    def test_linear_then_zero(self):
        gen = RampInput(1, 0.0 * u.nA, 10.0 * u.nA, duration=1.0 * u.ms)  # 10 steps
        out = self._run(gen, 12, u.nA)[:, 0]
        # value at step k (t=k*dt) is 10 * (t / 1ms) = k for k in [0, 10)
        npt.assert_allclose(out[:10], np.arange(10), atol=1e-5)
        npt.assert_allclose(out[10:], 0.0)

    def test_t_start_offset(self):
        gen = RampInput(1, 0.0 * u.nA, 10.0 * u.nA, duration=1.0 * u.ms, t_start=0.2 * u.ms)
        out = self._run(gen, 4, u.nA)[:, 0]
        npt.assert_allclose(out[:2], 0.0)            # before t_start
        npt.assert_allclose(out[2], 0.0, atol=1e-5)  # at t_start, frac=0


class TestStepInput(_AnalogInputTestBase):
    def test_staircase_holds_last(self):
        gen = StepInput(1, [0.0, 5.0, 10.0] * u.nA, [0.0, 0.3, 0.6] * u.ms)
        out = self._run(gen, 11, u.nA)[:, 0]
        npt.assert_allclose(out[0:3], 0.0)
        npt.assert_allclose(out[3:6], 5.0)
        npt.assert_allclose(out[6:11], 10.0)  # holds last amplitude past final step

    def test_zero_before_first_step(self):
        gen = StepInput(1, [5.0, 10.0] * u.nA, [0.2, 0.5] * u.ms)
        out = self._run(gen, 6, u.nA)[:, 0]
        npt.assert_allclose(out[0:2], 0.0)   # t < first step time
        npt.assert_allclose(out[2:5], 5.0)


class TestSectionInput(_AnalogInputTestBase):
    def test_piecewise_constant_then_zero(self):
        gen = SectionInput(1, [1.0, 2.0, 3.0] * u.nA, [0.2, 0.3, 0.2] * u.ms)
        out = self._run(gen, 8, u.nA)[:, 0]
        npt.assert_allclose(out[0:2], 1.0)   # [0, 0.2)
        npt.assert_allclose(out[2:5], 2.0)   # [0.2, 0.5)
        npt.assert_allclose(out[5:7], 3.0)   # [0.5, 0.7)
        npt.assert_allclose(out[7], 0.0)     # past total duration


class TestSinusoidalInput(_AnalogInputTestBase):
    def test_matches_closed_form(self):
        gen = SinusoidalInput(1, 2.0 * u.nA, 1000.0 * u.Hz, duration=2.0 * u.ms)
        out = self._run(gen, 15, u.nA)[:, 0]
        k = np.arange(15)
        t_s = k * 1e-4  # dt = 0.1 ms in seconds
        expected = 2.0 * np.sin(2 * np.pi * 1000.0 * t_s)
        expected[t_s >= 2e-3] = 0.0  # outside [0, duration)
        npt.assert_allclose(out, expected, atol=1e-5)

    def test_bias_offset(self):
        gen = SinusoidalInput(1, 2.0 * u.nA, 1000.0 * u.Hz, duration=2.0 * u.ms,
                              bias=1.0 * u.nA)
        out = self._run(gen, 10, u.nA)[:, 0]
        k = np.arange(10)
        expected = 1.0 + 2.0 * np.sin(2 * np.pi * 1000.0 * k * 1e-4)
        npt.assert_allclose(out, expected, atol=1e-5)


class TestWienerProcessInput(_AnalogInputTestBase):
    def test_shape_and_zero_outside_window(self):
        gen = WienerProcessInput(4, 2.0 * u.nA, duration=0.3 * u.ms)  # 3 steps
        with brainstate.environ.context(dt=self.dt):
            brainstate.random.seed(0)
        out = self._run(gen, 6, u.nA)
        self.assertEqual(out.shape, (6, 4))
        npt.assert_allclose(out[3:], 0.0)  # past duration

    def test_noise_scale(self):
        gen = WienerProcessInput(1, 2.0 * u.nA, duration=10000.0 * u.ms)
        with brainstate.environ.context(dt=self.dt):
            brainstate.random.seed(0)
        out = self._run(gen, 5000, u.nA)[:, 0]
        # increment std = sigma * sqrt(dt_in_ms) = 2 * sqrt(0.1) ≈ 0.632
        self.assertAlmostEqual(float(out.mean()), 0.0, delta=0.05)
        self.assertAlmostEqual(float(out.std()), 2.0 * np.sqrt(0.1), delta=0.05)

    def test_deterministic_with_seed(self):
        def run():
            gen = WienerProcessInput(2, 1.0 * u.nA, duration=1.0 * u.ms)
            with brainstate.environ.context(dt=self.dt):
                brainstate.random.seed(123)
            return self._run(gen, 5, u.nA)
        npt.assert_array_equal(run(), run())


class TestOUProcessInput(_AnalogInputTestBase):
    def test_zero_sigma_holds_mean(self):
        gen = OUProcessInput(3, 1.5 * u.nA, 0.0 * u.nA, 10.0 * u.ms, duration=0.4 * u.ms)
        out = self._run(gen, 6, u.nA)
        self.assertEqual(out.shape, (6, 3))
        npt.assert_allclose(out[:4], 1.5, atol=1e-5)  # sigma=0 => x stays at mean
        npt.assert_allclose(out[4:], 0.0)             # past duration

    def test_mean_reverting_bounded(self):
        gen = OUProcessInput(1, 0.0 * u.nA, 1.0 * u.nA, 5.0 * u.ms, duration=10000.0 * u.ms)
        with brainstate.environ.context(dt=self.dt):
            brainstate.random.seed(0)
        out = self._run(gen, 5000, u.nA)[:, 0]
        # The sample mean of a finite OU trajectory has a standard error of
        # ~sigma*sqrt(tau/2) / sqrt(T / (2*tau)) ~= 0.16 nA here, so assert mean
        # reversion with a ~3.5-SEM tolerance. A tighter bound is flaky: the RNG
        # stream (hence the exact mean) shifts with global float/x64 state that
        # other tests in the suite may set, even with a fixed seed.
        self.assertAlmostEqual(float(out.mean()), 0.0, delta=0.6)
        # steady-state std = sigma * sqrt(tau / 2) ~= 1.58 nA for this discretization:
        self.assertGreater(float(out.std()), 0.5)  # noise is present (not stuck at mean)
        self.assertLess(float(out.std()), 5.0)      # and the process stays bounded

    def test_deterministic_with_seed(self):
        def run():
            gen = OUProcessInput(2, 0.0 * u.nA, 1.0 * u.nA, 5.0 * u.ms, duration=1.0 * u.ms)
            with brainstate.environ.context(dt=self.dt):
                brainstate.random.seed(7)
            return self._run(gen, 5, u.nA)
        npt.assert_array_equal(run(), run())


class TestAnalogInputUnderJit(_AnalogInputTestBase):
    def test_ramp_runs_in_for_loop(self):
        gen = RampInput(2, 0.0 * u.nA, 10.0 * u.nA, duration=1.0 * u.ms)

        def step(i):
            with brainstate.environ.context(i=i):
                return gen.update()

        with brainstate.environ.context(dt=self.dt):
            brainstate.nn.init_all_states(gen)
            ys = brainstate.transform.for_loop(step, np.arange(10))
        self.assertEqual(ys.shape, (10, 2))
        npt.assert_allclose(np.asarray(ys.to_decimal(u.nA))[:, 0], np.arange(10), atol=1e-5)


if __name__ == '__main__':
    unittest.main()
