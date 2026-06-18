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
"""Regression tests for :mod:`brainpy.losses.comparison`.

These lock in the contract that the comparison losses which delegate to
``braintools.metric`` stay numerically identical to the upstream
implementation, while keeping brainpy's pytree / ``bm.Array`` envelope.
"""

import unittest

import jax
import numpy as np

import braintools.metric as M
import brainpy.math as bm
from brainpy import losses as L


def _arr(*shape, seed=0):
    return bm.asarray(np.random.RandomState(seed).randn(*shape))


def _close(a, b, atol=1e-5):
    return np.allclose(np.asarray(a), np.asarray(b), atol=atol)


class TestDelegatedEquivalence(unittest.TestCase):
    """Each delegated loss must equal its braintools.metric counterpart."""

    def setUp(self):
        rng = np.random.RandomState(7)
        self.pred = bm.asarray(rng.randn(4, 5))
        self.tar = bm.asarray(rng.randn(4, 5))
        self.logits = bm.asarray(rng.randn(4, 5))
        labels = rng.rand(4, 5)
        self.labels = bm.asarray(labels / labels.sum(-1, keepdims=True))

    def test_l1_loss(self):
        for red in ('mean', 'sum', 'none'):
            self.assertTrue(_close(L.l1_loss(self.pred, self.tar, reduction=red),
                                   M.l1_loss(self.pred, self.tar, reduction=red)))

    def test_l2_loss(self):
        self.assertTrue(_close(L.l2_loss(self.pred, self.tar),
                               M.l2_loss(self.pred, self.tar)))

    def test_huber_loss(self):
        for delta in (0.5, 1.0, 2.0):
            self.assertTrue(_close(L.huber_loss(self.pred, self.tar, delta=delta),
                                   M.huber_loss(self.pred, self.tar, delta=delta)))

    def test_softmax_cross_entropy(self):
        self.assertTrue(_close(L.softmax_cross_entropy(self.logits, self.labels),
                               M.softmax_cross_entropy(self.logits, self.labels)))

    def test_sigmoid_binary_cross_entropy(self):
        self.assertTrue(_close(L.sigmoid_binary_cross_entropy(self.logits, self.labels),
                               M.sigmoid_binary_cross_entropy(self.logits, self.labels)))

    def test_log_cosh_loss(self):
        self.assertTrue(_close(L.log_cosh_loss(self.pred, self.tar),
                               M.log_cosh(self.pred, self.tar)))

    def test_mean_squared_error(self):
        for red in ('mean', 'sum', 'none'):
            self.assertTrue(_close(L.mean_squared_error(self.pred, self.tar, reduction=red),
                                   M.squared_error(self.pred, self.tar, reduction=red)))

    def test_mean_squared_error_axis(self):
        self.assertTrue(_close(L.mean_squared_error(self.pred, self.tar, axis=-1, reduction='mean'),
                               M.squared_error(self.pred, self.tar, axis=-1, reduction='mean')))

    def test_mean_absolute_error(self):
        for red in ('mean', 'sum', 'none'):
            self.assertTrue(_close(L.mean_absolute_error(self.pred, self.tar, reduction=red),
                                   M.absolute_error(self.pred, self.tar, reduction=red)))


class TestReductionDefaults(unittest.TestCase):
    """Public default ``reduction`` values must not change."""

    def test_l1_loss_default_is_mean(self):
        # P1-M1 fix: the functional ``l1_loss`` default reduction is now
        # ``'mean'`` (matching the ``L1Loss`` class, the docstring and PyTorch),
        # not the surprising ``'sum'`` it previously defaulted to.
        pred, tar = _arr(3, 4, seed=1), _arr(3, 4, seed=2)
        self.assertTrue(_close(L.l1_loss(pred, tar),
                               L.l1_loss(pred, tar, reduction='mean')))
        # and it must equal the OO wrapper's default.
        self.assertTrue(_close(L.l1_loss(pred, tar),
                               L.L1Loss().update(pred, tar)))

    def test_mse_default_is_mean(self):
        pred, tar = _arr(3, 4, seed=1), _arr(3, 4, seed=2)
        self.assertEqual(np.asarray(L.mean_squared_error(pred, tar)).shape, ())

    def test_mae_default_is_mean(self):
        pred, tar = _arr(3, 4, seed=1), _arr(3, 4, seed=2)
        self.assertEqual(np.asarray(L.mean_absolute_error(pred, tar)).shape, ())


class TestPytreeAndArrayEnvelope(unittest.TestCase):
    """Delegation must keep pytree-input support and bm.Array acceptance."""

    def test_pytree_l2_loss(self):
        a, b = _arr(3, 4, seed=1), _arr(3, 4, seed=2)
        flat = L.l2_loss(a, b)
        tree = L.l2_loss({'x': a}, {'x': b})
        self.assertTrue(_close(flat, tree))

    def test_pytree_huber(self):
        a, b = _arr(3, 4, seed=3), _arr(3, 4, seed=4)
        flat = L.huber_loss(a, b)
        tree = L.huber_loss({'x': a}, {'x': b})
        self.assertTrue(_close(flat, tree))

    def test_pytree_multi_leaf_sums(self):
        a, b = _arr(3, 4, seed=5), _arr(3, 4, seed=6)
        # two identical leaves -> _multi_return sums them -> 2x single leaf
        single = L.mean_squared_error(a, b, reduction='sum')
        multi = L.mean_squared_error({'x': a, 'y': a}, {'x': b, 'y': b}, reduction='sum')
        self.assertTrue(_close(2.0 * np.asarray(single), multi))

    def test_accepts_bm_array(self):
        a, b = _arr(2, 3, seed=8), _arr(2, 3, seed=9)
        self.assertIsNotNone(L.l1_loss(a, b))


class TestCTC(unittest.TestCase):
    """ctc_loss / ctc_loss_with_forward_probs delegate to braintools.metric."""

    def setUp(self):
        rng = np.random.RandomState(0)
        self.B, self.T, self.K, self.N = 2, 6, 4, 3
        self.logits = bm.asarray(rng.randn(self.B, self.T, self.K))
        self.logit_pad = bm.asarray(np.zeros((self.B, self.T)))
        self.labels = bm.asarray(rng.randint(1, self.K, size=(self.B, self.N)))
        self.label_pad = bm.asarray(np.zeros((self.B, self.N)))

    def test_ctc_loss_matches(self):
        a = L.ctc_loss(self.logits, self.logit_pad, self.labels, self.label_pad)
        b = M.ctc_loss(self.logits, self.logit_pad, self.labels, self.label_pad)
        self.assertEqual(np.asarray(a).shape, (self.B,))
        self.assertTrue(_close(a, b))

    def test_ctc_forward_probs_matches(self):
        la, pa, na = L.ctc_loss_with_forward_probs(
            self.logits, self.logit_pad, self.labels, self.label_pad)
        lb, pb, nb = M.ctc_loss_with_forward_probs(
            self.logits, self.logit_pad, self.labels, self.label_pad)
        self.assertEqual(np.asarray(la).shape, (self.B,))
        self.assertTrue(_close(la, lb))
        self.assertTrue(_close(pa, pb))
        self.assertTrue(_close(na, nb))


class TestUntouchedLosses(unittest.TestCase):
    """Losses with no safe braintools equivalent keep working unchanged."""

    def test_cross_entropy_loss_runs(self):
        rng = np.random.RandomState(2)
        logits = bm.asarray(rng.randn(4, 5))
        targets = bm.asarray(rng.randint(0, 5, size=(4,)))
        self.assertEqual(np.asarray(L.cross_entropy_loss(logits, targets)).shape, ())

    def test_cross_entropy_sparse_runs(self):
        rng = np.random.RandomState(2)
        logits = bm.asarray(rng.randn(4, 5))
        targets = bm.asarray(rng.randint(0, 5, size=(4, 1)))
        self.assertEqual(np.asarray(L.cross_entropy_sparse(logits, targets)).shape, (4,))

    def test_multi_margin_loss_runs(self):
        rng = np.random.RandomState(2)
        # multi_margin_loss indexes ``predicts`` by ``targets``; use plain
        # jax/numpy inputs (the supported indexing path) since this function
        # is intentionally left untouched.
        logits = bm.as_jax(bm.asarray(rng.randn(4, 5)))
        targets = rng.randint(0, 5, size=(4,))
        self.assertIsNotNone(L.multi_margin_loss(logits, targets))


class TestMultiMarginArrayEnvelope(unittest.TestCase):
    """P1-H2: ``multi_margin_loss`` must accept ``bm.Array`` inputs.

    Under JAX >= 0.9 implicit ``__jax_array__`` coercion was removed, so
    indexing a ``bm.Array`` with ``jnp`` advanced indexing raised
    ``ValueError: Triggering __jax_array__() ... no longer supported``. Every
    other loss in the module accepts ``bm.Array``; this one must too.
    """

    def test_multi_margin_accepts_bm_array(self):
        predicts = bm.asarray(np.array([[0.2, 0.8], [0.6, 0.4]]))
        targets = bm.asarray(np.array([1, 0]))
        out = L.multi_margin_loss(predicts, targets, margin=1.0, p=1, reduction='mean')
        self.assertTrue(np.isfinite(float(out)))

    def test_multi_margin_bm_matches_jax(self):
        p_np = np.array([[0.2, 0.8, 0.1], [0.6, 0.4, 0.9]])
        t_np = np.array([1, 0])
        bm_out = np.asarray(L.multi_margin_loss(bm.asarray(p_np), bm.asarray(t_np),
                                                p=2, reduction='none'))
        jax_out = np.asarray(L.multi_margin_loss(bm.as_jax(bm.asarray(p_np)),
                                                 t_np, p=2, reduction='none'))
        self.assertTrue(_close(bm_out, jax_out))


if __name__ == '__main__':
    unittest.main()
