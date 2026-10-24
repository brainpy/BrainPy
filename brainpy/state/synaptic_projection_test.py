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
from brainpy.state import IF, SymmetryGapJunction, AsymmetryGapJunction


def make_conn(pre_ids, post_ids):
    """Create a connection function returning fixed pre/post indices."""
    pre_ids = jnp.array(pre_ids)
    post_ids = jnp.array(post_ids)
    return lambda pre_size, post_size: (pre_ids, post_ids)


class TestSymmetryGapJunction(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.n = 10
        self.batch_size = 4
        self.dt = 0.1 * u.ms

    def test_self_coupling_init(self):
        neurons = IF(self.n)
        neurons.init_state(self.batch_size)
        conn = make_conn([0, 1, 2], [1, 2, 0])
        gj = SymmetryGapJunction(
            couples=neurons, states='V', conn=conn,
            weight=0.1 * u.mS,
        )
        self.assertIs(gj.pre, gj.post)
        self.assertEqual(gj.pre_state, 'V')
        self.assertEqual(gj.post_state, 'V')
        self.assertEqual(len(gj.pre_ids), 3)
        self.assertEqual(len(gj.post_ids), 3)

    def test_two_population_init(self):
        pre = IF(self.n)
        post = IF(self.n)
        pre.init_state(self.batch_size)
        post.init_state(self.batch_size)
        conn = make_conn([0, 1], [2, 3])
        gj = SymmetryGapJunction(
            couples=(pre, post), states=('V', 'V'), conn=conn,
            weight=0.5 * u.mS,
        )
        self.assertIsNot(gj.pre, gj.post)
        self.assertEqual(gj.pre_state, 'V')
        self.assertEqual(gj.post_state, 'V')

    def test_forward_pass_batched(self):
        neurons = IF(self.n)
        neurons.init_state(self.batch_size)
        conn = make_conn([0, 1, 2], [1, 2, 0])
        gj = SymmetryGapJunction(
            couples=neurons, states='V', conn=conn,
            weight=0.1 * u.mS,
        )
        with brainstate.environ.context(dt=self.dt):
            result = gj.update()
        self.assertEqual(result.shape, (self.batch_size, self.n))

    def test_forward_pass_unbatched(self):
        neurons = IF(self.n)
        neurons.init_state()
        conn = make_conn([0, 1], [1, 2])
        gj = SymmetryGapJunction(
            couples=neurons, states='V', conn=conn,
            weight=0.1 * u.mS,
        )
        with brainstate.environ.context(dt=self.dt):
            result = gj.update()
        self.assertEqual(result.shape, (self.n,))

    def test_equal_voltages_zero_current(self):
        """When all voltages are equal, gap junction current should be zero."""
        neurons = IF(self.n)
        neurons.init_state(self.batch_size)
        # All V = 0 mV by default, so no voltage difference
        conn = make_conn([0, 1], [1, 2])
        gj = SymmetryGapJunction(
            couples=neurons, states='V', conn=conn,
            weight=0.1 * u.mS,
        )
        with brainstate.environ.context(dt=self.dt):
            result = gj.update()
        # Result should be all zeros (zero voltage diff * weight)
        self.assertTrue(jnp.allclose(u.get_magnitude(result), 0.0))

    def test_nonzero_voltage_diff(self):
        """Voltage differences should produce non-zero currents."""
        neurons = IF(self.n)
        neurons.init_state(self.batch_size)
        neurons.V.value = neurons.V.value.at[..., 0].set(10.0 * u.mV)
        conn = make_conn([0], [1])
        gj = SymmetryGapJunction(
            couples=neurons, states='V', conn=conn,
            weight=0.1 * u.mS,
        )
        with brainstate.environ.context(dt=self.dt):
            result = gj.update()
        self.assertFalse(jnp.allclose(u.get_magnitude(result), 0.0))

    def test_weight_shape(self):
        """Weight should be stored as a parameter."""
        neurons = IF(self.n)
        neurons.init_state(self.batch_size)
        conn = make_conn([0, 1, 2], [3, 4, 5])
        # Per-connection weight array
        weight = jnp.array([0.1, 0.2, 0.3]) * u.mS
        gj = SymmetryGapJunction(
            couples=neurons, states='V', conn=conn,
            weight=weight,
        )
        self.assertEqual(gj.weight.value.shape, (3,))

    def test_adds_current_inputs(self):
        """Update should add current inputs to both pre and post."""
        pre = IF(self.n)
        post = IF(self.n)
        pre.init_state(self.batch_size)
        post.init_state(self.batch_size)
        pre.V.value = pre.V.value.at[..., 0].set(5.0 * u.mV)
        conn = make_conn([0], [0])
        gj = SymmetryGapJunction(
            couples=(pre, post), states=('V', 'V'), conn=conn,
            weight=0.1 * u.mS,
        )
        with brainstate.environ.context(dt=self.dt):
            gj.update()
        # Both pre and post should have received current inputs
        self.assertIsNotNone(pre.current_inputs)
        self.assertIsNotNone(post.current_inputs)
        self.assertGreater(len(pre.current_inputs), 0)
        self.assertGreater(len(post.current_inputs), 0)

    def test_input_key_stable_independent_of_other_inputs(self):
        """Regression: the gap-junction input key must be stable per instance and
        must not drift as unrelated inputs are registered on the target. A
        count-derived key leaked a fresh slot (``gap_junction_post_0``, ``_2`` ...)
        on every un-consumed update, silently accumulating stale currents."""
        pre = IF(self.n)
        post = IF(self.n)
        pre.init_state()
        post.init_state()
        conn = make_conn([0], [0])
        gj = SymmetryGapJunction(
            couples=(pre, post), states=('V', 'V'), conn=conn,
            weight=0.1 * u.mS,
        )

        def external(*args, **kwargs):
            return u.math.zeros(post.varshape) * u.mA

        # Unrelated persistent input already present on the target.
        post.add_current_input('external_1', external)
        with brainstate.environ.context(dt=self.dt):
            gj.update()
        keys_1 = sorted(k for k in post.current_inputs if k.startswith('gap_junction_post'))
        # Consume both sides, as the per-step network loop does: pops the
        # gap-junction array inputs, keeps the persistent callable.
        post.sum_current_inputs(u.math.zeros(post.varshape) * u.mA)
        pre.sum_current_inputs(u.math.zeros(pre.varshape) * u.mA)
        # Register another unrelated input, shifting any positional index.
        post.add_current_input('external_2', external)
        with brainstate.environ.context(dt=self.dt):
            gj.update()
        keys_2 = sorted(k for k in post.current_inputs if k.startswith('gap_junction_post'))

        self.assertEqual(len(keys_1), 1)
        self.assertEqual(keys_1, keys_2)


class TestAsymmetryGapJunction(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)
        self.n = 10
        self.batch_size = 4
        self.dt = 0.1 * u.ms

    def test_init(self):
        pre = IF(self.n)
        post = IF(self.n)
        pre.init_state(self.batch_size)
        post.init_state(self.batch_size)
        conn = make_conn([0, 1], [1, 2])
        weight = jnp.ones((2, 2)) * u.mS
        gj = AsymmetryGapJunction(
            pre=pre, pre_state='V', post=post, post_state='V',
            conn=conn, weight=weight,
        )
        self.assertEqual(gj.weight.value.shape, (2, 2))
        self.assertEqual(gj.pre_state, 'V')
        self.assertEqual(gj.post_state, 'V')

    def test_forward_pass_batched(self):
        pre = IF(self.n)
        post = IF(self.n)
        pre.init_state(self.batch_size)
        post.init_state(self.batch_size)
        conn = make_conn([0, 1, 2], [3, 4, 5])
        weight = jnp.ones((3, 2)) * u.mS
        gj = AsymmetryGapJunction(
            pre=pre, pre_state='V', post=post, post_state='V',
            conn=conn, weight=weight,
        )
        with brainstate.environ.context(dt=self.dt):
            result = gj.update()
        self.assertEqual(result.shape, (self.batch_size, self.n))

    def test_forward_pass_unbatched(self):
        pre = IF(self.n)
        post = IF(self.n)
        pre.init_state()
        post.init_state()
        conn = make_conn([0], [1])
        weight = jnp.array([0.1, 0.2]) * u.mS
        gj = AsymmetryGapJunction(
            pre=pre, pre_state='V', post=post, post_state='V',
            conn=conn, weight=weight,
        )
        with brainstate.environ.context(dt=self.dt):
            result = gj.update()
        self.assertEqual(result.shape, (self.n,))

    def test_1d_weight(self):
        """1D weight array [pre_w, post_w] applies same weights to all connections."""
        pre = IF(self.n)
        post = IF(self.n)
        pre.init_state(self.batch_size)
        post.init_state(self.batch_size)
        pre.V.value = pre.V.value.at[..., 0].set(10.0 * u.mV)
        conn = make_conn([0], [1])
        weight = jnp.array([0.1, 0.3]) * u.mS
        gj = AsymmetryGapJunction(
            pre=pre, pre_state='V', post=post, post_state='V',
            conn=conn, weight=weight,
        )
        with brainstate.environ.context(dt=self.dt):
            result = gj.update()
        self.assertEqual(result.shape, (self.batch_size, self.n))
        self.assertFalse(jnp.allclose(u.get_magnitude(result), 0.0))

    def test_2d_weight(self):
        """2D weight array of shape (n_connections, 2) for per-connection weights."""
        pre = IF(self.n)
        post = IF(self.n)
        pre.init_state(self.batch_size)
        post.init_state(self.batch_size)
        conn = make_conn([0, 1, 2], [3, 4, 5])
        weight = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]) * u.mS
        gj = AsymmetryGapJunction(
            pre=pre, pre_state='V', post=post, post_state='V',
            conn=conn, weight=weight,
        )
        self.assertEqual(gj.weight.value.shape, (3, 2))

    def test_equal_voltages_zero_current(self):
        pre = IF(self.n)
        post = IF(self.n)
        pre.init_state(self.batch_size)
        post.init_state(self.batch_size)
        conn = make_conn([0, 1], [1, 2])
        weight = jnp.ones((2, 2)) * u.mS
        gj = AsymmetryGapJunction(
            pre=pre, pre_state='V', post=post, post_state='V',
            conn=conn, weight=weight,
        )
        with brainstate.environ.context(dt=self.dt):
            result = gj.update()
        self.assertTrue(jnp.allclose(u.get_magnitude(result), 0.0))

    def test_adds_current_inputs(self):
        pre = IF(self.n)
        post = IF(self.n)
        pre.init_state(self.batch_size)
        post.init_state(self.batch_size)
        pre.V.value = pre.V.value.at[..., 0].set(5.0 * u.mV)
        conn = make_conn([0], [0])
        weight = jnp.array([0.1, 0.2]) * u.mS
        gj = AsymmetryGapJunction(
            pre=pre, pre_state='V', post=post, post_state='V',
            conn=conn, weight=weight,
        )
        with brainstate.environ.context(dt=self.dt):
            gj.update()
        self.assertIsNotNone(pre.current_inputs)
        self.assertIsNotNone(post.current_inputs)

    def test_input_key_stable_independent_of_other_inputs(self):
        """Regression: the gap-junction input key must be stable per instance and
        must not drift as unrelated inputs are registered on the target."""
        pre = IF(self.n)
        post = IF(self.n)
        pre.init_state()
        post.init_state()
        conn = make_conn([0], [0])
        weight = jnp.array([0.1, 0.2]) * u.mS
        gj = AsymmetryGapJunction(
            pre=pre, pre_state='V', post=post, post_state='V',
            conn=conn, weight=weight,
        )

        def external(*args, **kwargs):
            return u.math.zeros(post.varshape) * u.mA

        post.add_current_input('external_1', external)
        with brainstate.environ.context(dt=self.dt):
            gj.update()
        keys_1 = sorted(k for k in post.current_inputs if k.startswith('gap_junction_post'))
        post.sum_current_inputs(u.math.zeros(post.varshape) * u.mA)
        pre.sum_current_inputs(u.math.zeros(pre.varshape) * u.mA)
        post.add_current_input('external_2', external)
        with brainstate.environ.context(dt=self.dt):
            gj.update()
        keys_2 = sorted(k for k in post.current_inputs if k.startswith('gap_junction_post'))

        self.assertEqual(len(keys_1), 1)
        self.assertEqual(keys_1, keys_2)


class TestGapJunctionMissingState(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(dt=0.1 * u.ms)

    def test_symmetry_missing_state_raises(self):
        neurons = IF(5)
        neurons.init_state()
        conn = make_conn([0], [1])
        gj = SymmetryGapJunction(
            couples=neurons, states='nonexistent', conn=conn,
            weight=0.1 * u.mS,
        )
        with self.assertRaises(ValueError):
            gj.update()

    def test_asymmetry_missing_state_raises(self):
        pre = IF(5)
        post = IF(5)
        pre.init_state()
        post.init_state()
        conn = make_conn([0], [1])
        weight = jnp.array([0.1, 0.2]) * u.mS
        gj = AsymmetryGapJunction(
            pre=pre, pre_state='nonexistent', post=post, post_state='V',
            conn=conn, weight=weight,
        )
        with self.assertRaises(ValueError):
            gj.update()


if __name__ == '__main__':
    unittest.main()
