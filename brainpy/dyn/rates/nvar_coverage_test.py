# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
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
"""Line-coverage tests for ``brainpy/dyn/rates/nvar.py`` (the NVAR feature map).

Exercises:

* the ``_comb`` helper across its ``N > k`` / ``N == k`` / ``N < k`` branches,
* construction with ``order=None`` (default), a single int order, and a
  sequence of multiple orders, including ``stride`` and ``constant`` options,
* output-dimension bookkeeping (``linear_dim``, ``nonlinear_dim``, ``num_out``),
* ``reset_state`` in both non-batching and batching modes,
* ``update`` for both the non-batching and batching code paths,
* ``get_feature_names`` with ``for_plot`` True/False and with a constant term,
* the constructor error branches (order < 2; non-boolean ``constant``).
"""

import jax.numpy as jnp
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm
from brainpy.dyn.rates.nvar import NVAR, _comb


class TestComb(parameterized.TestCase):
    def test_comb_branches(self):
        self.assertEqual(_comb(5, 2), 10)   # N > k
        self.assertEqual(_comb(3, 3), 1)    # N == k
        self.assertEqual(_comb(2, 5), 0)    # N < k


class TestNVARConstruction(parameterized.TestCase):
    def test_order_none_default(self):
        node = NVAR(num_in=3, delay=2)
        self.assertEqual(node.order, tuple())
        # no nonlinear terms when there is no order
        self.assertEqual(node.nonlinear_dim, 0)
        self.assertEqual(node.num_out, node.linear_dim)

    def test_single_int_order_wrapped(self):
        node = NVAR(num_in=3, delay=2, order=2)
        self.assertEqual(node.order, (2,))

    def test_multi_order_stride_constant(self):
        node = NVAR(num_in=3, delay=4, order=[2, 3], stride=2, constant=True)
        # num_delay = 1 + (delay - 1) * stride
        self.assertEqual(node.num_delay, 1 + (4 - 1) * 2)
        self.assertEqual(node.linear_dim, 4 * 3)
        self.assertEqual(len(node.comb_ids), 2)
        # +1 for the constant column
        self.assertEqual(node.num_out, node.linear_dim + node.nonlinear_dim + 1)


class TestNVARErrors(parameterized.TestCase):
    def test_order_below_two_raises(self):
        with self.assertRaises(AssertionError):
            NVAR(num_in=3, delay=2, order=1)

    def test_non_bool_constant_raises(self):
        with self.assertRaises(AssertionError):
            NVAR(num_in=3, delay=2, constant=1)


class TestNVARUpdate(parameterized.TestCase):
    def test_nonbatching_update_and_names(self):
        node = NVAR(num_in=3, delay=4, order=[2, 3], stride=2, constant=True)
        out = node.update(bm.ones(3))
        self.assertTupleEqual(tuple(out.shape), (node.num_out,))

        names = node.get_feature_names()
        self.assertEqual(len(names), node.num_out)
        self.assertEqual(names[0], '1')           # constant term first

        plot_names = node.get_feature_names(for_plot=True)
        self.assertEqual(len(plot_names), node.num_out)

        # reset_state for the non-batching branch (batch_or_mode is None)
        node.reset_state()
        self.assertTupleEqual(tuple(node.store.shape), (node.num_delay, node.num_in))
        self.assertEqual(int(node.idx[0]), 0)

    def test_nonbatching_no_constant_names(self):
        node = NVAR(num_in=2, delay=3, order=2, constant=False)
        names = node.get_feature_names()
        self.assertNotEqual(names[0], '1')
        self.assertEqual(len(names), node.num_out)

    def test_batching_update(self):
        node = NVAR(num_in=2, delay=3, order=2, constant=True,
                    mode=bm.BatchingMode())
        node.reset_state(batch_or_mode=4)
        self.assertTupleEqual(tuple(node.store.shape), (node.num_delay, 4, node.num_in))
        out = node.update(bm.ones((4, 2)))
        self.assertEqual(out.shape[0], 4)
        self.assertEqual(out.shape[1], node.num_out)

    def test_layer_call_matches_existing(self):
        bm.random.seed()
        inp = bm.random.randn(1, 5)
        node = NVAR(num_in=5, delay=10, mode=bm.BatchingMode())
        out = node(inp)
        self.assertEqual(out.shape[-1], node.num_out)


if __name__ == '__main__':
    parameterized.absltest.main()
