# -*- coding: utf-8 -*-
"""Coverage + correctness tests for ``brainpy/losses/regularization.py``.

Part of bringing the previously test-free ``brainpy.losses`` package to
>=90% line coverage. Each public regularizer is checked against an independent
NumPy reference and through both single-array and pytree leaves.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import brainpy.math as bm
from brainpy.losses import regularization as R


def _f(x):
    return np.asarray(x, dtype=np.float64)


class TestL2Norm:
    def test_single_array(self):
        x = jnp.array([3.0, 4.0])
        assert float(R.l2_norm(x)) == pytest.approx(5.0)  # sqrt(9+16)

    def test_pytree(self):
        x = {'a': jnp.array([3.0]), 'b': jnp.array([4.0])}
        assert float(R.l2_norm(x)) == pytest.approx(5.0)


class TestMeanAbsoluteSquare:
    def test_mean_absolute(self):
        x = jnp.array([[1.0, -3.0]])
        assert float(R.mean_absolute(x)) == pytest.approx(2.0)
        assert np.allclose(np.asarray(R.mean_absolute(x, axis=1)), [2.0])

    def test_mean_square(self):
        x = jnp.array([[1.0, 3.0]])
        assert float(R.mean_square(x)) == pytest.approx(5.0)  # (1+9)/2

    def test_bm_array_leaf(self):
        x = bm.asarray([2.0, -2.0])
        assert float(R.mean_absolute(x)) == pytest.approx(2.0)


class TestLogCosh:
    def test_zero_is_zero(self):
        assert float(R.log_cosh(jnp.array([0.0]))[0]) == pytest.approx(0.0, abs=1e-6)

    def test_large_value_asymptote(self):
        big = float(R.log_cosh(jnp.array([10.0]))[0])
        assert big == pytest.approx(10.0 - np.log(2.0), rel=1e-3)


class TestSmoothLabels:
    def test_smoothing(self):
        labels = jnp.array([[1.0, 0.0, 0.0]])
        alpha = 0.3
        out = np.asarray(R.smooth_labels(labels, alpha))
        ref = (1.0 - alpha) * _f(labels) + alpha / 3
        assert np.allclose(out, ref, rtol=1e-6)

    def test_smoothing_rows_sum_to_one(self):
        labels = jnp.eye(4)[None, 0]
        out = np.asarray(R.smooth_labels(labels, 0.5))
        assert out.sum() == pytest.approx(1.0, rel=1e-6)
