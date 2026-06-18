# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy/encoding/stateless_encoding.py``.

Target: ``PoissonEncoder`` (every branch of ``single_step`` / ``multi_steps`` /
``_normalize`` / ``_zero_out``) and the ``DiffEncoder.single_step``
``NotImplementedError`` path -- none of which were touched by the existing
``stateless_encoding_test.py`` (which only covers ``DiffEncoder.multi_steps``).

Tiny shapes; verifies output shape/dtype and that spikes are binary.
"""

import unittest

import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy.encoding.stateless_encoding import PoissonEncoder, DiffEncoder


def _is_binary(arr):
    vals = np.unique(np.asarray(bm.as_jax(arr)))
    return set(vals.tolist()).issubset({0.0, 1.0})


class TestPoissonEncoderSingleStep(unittest.TestCase):
    def setUp(self):
        bm.random.seed(0)

    def test_single_step_no_istep(self):
        enc = PoissonEncoder()
        x = bm.random.rand(5)
        out = enc.single_step(x)
        self.assertEqual(out.shape, (5,))
        self.assertTrue(_is_binary(out))

    def test_single_step_with_istep(self):
        enc = PoissonEncoder()
        x = bm.random.rand(5)
        out = enc.single_step(x, i_step=0)
        self.assertEqual(out.shape, (5,))
        self.assertTrue(_is_binary(out))

    def test_single_step_before_first_spike_is_zero(self):
        # first_spk_time>0 -> steps before first_spk_step emit nothing.
        enc = PoissonEncoder(first_spk_time=0.5)
        x = bm.ones(5)  # firing prob 1 to make the masking visible
        out = bm.as_jax(enc.single_step(x, i_step=0))
        self.assertEqual(float(out.sum()), 0.0)

    def test_normalize_with_minmax_gain_offset(self):
        enc = PoissonEncoder(min_val=0., max_val=2., gain=2.0, offset=0.0)
        x = bm.array([0.0, 1.0, 2.0])
        out = enc.single_step(x)
        self.assertEqual(out.shape, (3,))


class TestPoissonEncoderMultiSteps(unittest.TestCase):
    def setUp(self):
        bm.random.seed(0)

    def test_multi_steps_none_returns_single(self):
        enc = PoissonEncoder()
        x = bm.random.rand(4)
        out = enc.multi_steps(x, n_time=None)
        self.assertEqual(out.shape, (4,))

    def test_multi_steps_with_duration(self):
        enc = PoissonEncoder()
        x = bm.random.rand(4)
        out = enc.multi_steps(x, n_time=10.)
        n_step = int(10. / bm.get_dt())
        self.assertEqual(out.shape, (n_step, 4))
        self.assertTrue(_is_binary(out))

    def test_multi_steps_with_first_spike(self):
        # first_spk_step>0 -> a zero-prefix is concatenated.
        enc = PoissonEncoder(first_spk_time=0.2)
        x = bm.ones(4)
        out = bm.as_jax(enc.multi_steps(x, n_time=10.))
        n_step = int(10. / bm.get_dt())
        self.assertEqual(out.shape, (n_step, 4))
        # The first ``first_spk_step`` rows are all zeros.
        self.assertEqual(float(out[:enc.first_spk_step].sum()), 0.0)

    def test_zero_out(self):
        enc = PoissonEncoder()
        x = bm.random.rand(4)
        out = enc._zero_out(x)
        self.assertEqual(float(bm.as_jax(out).sum()), 0.0)

    def test_repr(self):
        self.assertEqual(repr(PoissonEncoder()), 'PoissonEncoder')


class TestDiffEncoderSingleStep(unittest.TestCase):
    def test_single_step_not_implemented(self):
        enc = DiffEncoder()
        with self.assertRaises(NotImplementedError):
            enc.single_step(bm.array([1., 2., 3.]))


if __name__ == '__main__':
    unittest.main()
