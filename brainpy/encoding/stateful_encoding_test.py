# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy/encoding/stateful_encoding.py``.

Target: ``WeightedPhaseEncoder`` (previously completely untested) and the
parts of ``LatencyEncoder`` not exercised by ``stateless_encoding_test.py``
(the ``log`` method, the ``clip`` branch, the ``n_time=None`` default, the
constructor validation branches, and ``single_step`` raising).

Tiny shapes; verifies output shape + structural spike-train properties.
"""

import unittest

import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy.encoding.stateful_encoding import WeightedPhaseEncoder, LatencyEncoder


class TestWeightedPhaseEncoder(unittest.TestCase):
    def setUp(self):
        bm.random.seed(0)

    def test_default_weight_fun(self):
        enc = WeightedPhaseEncoder(min_val=0., max_val=1., num_phase=8)
        x = bm.array([0.1, 0.5, 0.9])
        out = enc(x, num_step=8)
        self.assertEqual(out.shape, (8, 3))
        # Output is a binary spike train.
        vals = np.unique(np.asarray(bm.as_jax(out)))
        self.assertTrue(set(vals.tolist()).issubset({0.0, 1.0}))

    def test_custom_weight_fun(self):
        enc = WeightedPhaseEncoder(0., 1., 4, weight_fun=lambda i: 2.0 ** (-(i % 4 + 1)))
        x = bm.array([0.25, 0.75])
        out = enc(x, num_step=4)
        self.assertEqual(out.shape, (4, 2))

    def test_repr(self):
        enc = WeightedPhaseEncoder(0., 1., 8)
        self.assertEqual(repr(enc), 'WeightedPhaseEncoder')

    def test_bad_num_phase_raises(self):
        with self.assertRaises(Exception):
            WeightedPhaseEncoder(0., 1., 0)


class TestLatencyEncoderConstructor(unittest.TestCase):
    def test_bad_method_raises(self):
        with self.assertRaises(ValueError):
            LatencyEncoder(method='bogus')

    def test_threshold_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            LatencyEncoder(threshold=-0.1)
        with self.assertRaises(ValueError):
            LatencyEncoder(threshold=1.1)

    def test_single_step_not_implemented(self):
        enc = LatencyEncoder(method='linear')
        with self.assertRaises(NotImplementedError):
            enc.single_step(bm.array([0.5]))


class TestLatencyEncoderMethods(unittest.TestCase):
    def setUp(self):
        bm.random.seed(0)

    def test_log_method(self):
        enc = LatencyEncoder(method='log')
        x = bm.array([0.1, 0.5, 0.9])
        out = enc.multi_steps(x, n_time=1.0)
        n_step = int(1.0 / bm.get_dt())
        self.assertEqual(out.shape, (n_step, 3))

    def test_log_method_with_minmax_normalize(self):
        enc = LatencyEncoder(min_val=0., max_val=2., method='log')
        x = bm.array([0.2, 1.0, 1.8])
        out = enc.multi_steps(x, n_time=1.0)
        self.assertEqual(out.shape[1], 3)

    def test_linear_method_with_minmax(self):
        enc = LatencyEncoder(min_val=0., max_val=1., method='linear')
        x = bm.array([0.0, 0.5, 1.0])
        out = enc.multi_steps(x, n_time=1.0)
        self.assertEqual(out.shape[1], 3)

    def test_clip_branch(self):
        # Values below threshold get pushed to +inf spike time; one_hot of an
        # out-of-range index produces an all-zero column.
        enc = LatencyEncoder(method='linear', clip=True, threshold=0.5)
        x = bm.array([0.1, 0.6, 0.9])
        out = bm.as_jax(enc.multi_steps(x, n_time=1.0))
        # The sub-threshold feature (index 0) should never spike.
        self.assertEqual(float(out[:, 0].sum()), 0.0)

    def test_n_time_none_uses_tau(self):
        enc = LatencyEncoder(method='log', tau=1.0)
        x = bm.array([0.2, 0.8])
        out = enc.multi_steps(x)  # n_time defaults to tau
        n_step = int(1.0 / bm.get_dt())
        self.assertEqual(out.shape, (n_step, 2))

    def test_normalize_uses_n_time_as_tau(self):
        enc = LatencyEncoder(method='linear', normalize=True)
        x = bm.array([0.02, 0.5, 1.0])
        out = enc.multi_steps(x, n_time=0.5)
        self.assertEqual(out.shape[1], 3)


if __name__ == '__main__':
    unittest.main()
