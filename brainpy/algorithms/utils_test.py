# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy.algorithms.utils``.

Exercises the small helper building blocks used by the offline/online
training algorithms:

* ``Sigmoid`` forward + gradient,
* the ``Regularization`` family (base no-op, L1, L2, L1L2) call + grad,
* ``index_combinations`` / ``polynomial_features`` (with and without bias,
  and the degree==1 short-circuit branch),
* ``normalize`` including the zero-norm guard branch.

Tiny hand-checked data is used so the numerical expectations are exact.
"""

import numpy as np

import brainpy.math as bm
from brainpy.algorithms import utils


class TestSigmoid:
    def test_forward_matches_logistic(self):
        s = utils.Sigmoid()
        x = bm.asarray([0.0, 2.0, -2.0])
        out = bm.as_jax(s(x))
        expected = 1.0 / (1.0 + np.exp(-np.array([0.0, 2.0, -2.0])))
        assert np.allclose(np.asarray(out), expected, atol=1e-6)
        # sigmoid(0) == 0.5 exactly
        assert np.allclose(np.asarray(out)[0], 0.5)

    def test_grad_matches_analytic(self):
        s = utils.Sigmoid()
        x = bm.asarray([0.0, 1.0, -1.0])
        g = bm.as_jax(s.grad(x))
        # d/dx sigmoid = sigmoid*(1-sigmoid); at 0 -> 0.25
        assert np.allclose(np.asarray(g)[0], 0.25, atol=1e-6)
        # symmetric around 0
        assert np.allclose(np.asarray(g)[1], np.asarray(g)[2], atol=1e-6)


class TestRegularization:
    def test_base_regularization_is_noop(self):
        reg = utils.Regularization(alpha=0.7)
        assert reg.alpha == 0.7
        w = bm.asarray([1.0, -2.0, 3.0])
        assert reg(w) == 0
        assert reg.grad(w) == 0

    def test_l1_regularization(self):
        reg = utils.L1Regularization(alpha=2.0)
        w = bm.asarray([3.0, -4.0])  # norm == 5
        # __call__ returns alpha * ||w||_2
        assert np.allclose(float(bm.as_jax(reg(w))), 2.0 * 5.0, atol=1e-6)
        # grad returns alpha * sign(w)
        g = np.asarray(bm.as_jax(reg.grad(w)))
        assert np.allclose(g, np.array([2.0, -2.0]), atol=1e-6)

    def test_l2_regularization(self):
        reg = utils.L2Regularization(alpha=2.0)
        w = bm.asarray([3.0, 4.0])
        # alpha * 0.5 * w.w == 2 * 0.5 * 25 == 25
        assert np.allclose(float(bm.as_jax(reg(w))), 25.0, atol=1e-5)
        g = np.asarray(bm.as_jax(reg.grad(w)))
        assert np.allclose(g, np.array([6.0, 8.0]), atol=1e-6)

    def test_l1l2_regularization(self):
        reg = utils.L1L2Regularization(alpha=1.0, l1_ratio=0.5)
        w = bm.asarray([3.0, 4.0])  # ||w||_2 == 5, w.w == 25
        # l1_contr = 0.5*5 = 2.5; l2_contr = 0.5*0.5*25 = 6.25; sum*alpha
        assert np.allclose(float(bm.as_jax(reg(w))), 8.75, atol=1e-5)
        g = np.asarray(bm.as_jax(reg.grad(w)))
        # 0.5*sign + 0.5*w
        assert np.allclose(g, np.array([0.5 * 1 + 0.5 * 3, 0.5 * 1 + 0.5 * 4]), atol=1e-6)


class TestPolynomialFeatures:
    def test_index_combinations_degree2(self):
        # for 2 features, degree 2 -> combos of size 2: (0,0),(0,1),(1,1)
        combs = utils.index_combinations(n_features=2, degree=2)
        assert (0, 0) in combs and (0, 1) in combs and (1, 1) in combs
        assert len(combs) == 3

    def test_degree1_shortcircuit_with_bias(self):
        # degree == 1 -> no combinations; with add_bias a 1-column is prepended
        X = bm.asarray([[2.0], [3.0]])
        out = np.asarray(bm.as_jax(utils.polynomial_features(X, degree=1, add_bias=True)))
        assert out.shape == (2, 2)
        assert np.allclose(out[:, 0], 1.0)
        assert np.allclose(out[:, 1], np.array([2.0, 3.0]))

    def test_degree1_shortcircuit_without_bias(self):
        X = bm.asarray([[2.0], [3.0]])
        out = np.asarray(bm.as_jax(utils.polynomial_features(X, degree=1, add_bias=False)))
        assert np.allclose(out, np.array([[2.0], [3.0]]))

    def test_degree2_with_bias(self):
        X = bm.asarray([[2.0, 3.0]])
        out = np.asarray(bm.as_jax(utils.polynomial_features(X, degree=2, add_bias=True)))
        # P16-M2: width is exactly 1 bias + 2 linear + 3 interaction == 6
        # (previously a dead all-zero trailing column made it 7).
        assert out.shape == (1, 6)
        assert out[0, 0] == 1.0
        # the linear features should appear
        assert 2.0 in out[0] and 3.0 in out[0]
        # interactions: 4 (=2^2), 6 (=2*3), 9 (=3^2)
        for v in (4.0, 6.0, 9.0):
            assert np.any(np.isclose(out[0], v))
        # no dead all-zero column anymore
        assert not np.any(np.all(out == 0, axis=0))

    def test_degree2_without_bias(self):
        X = bm.asarray([[2.0, 3.0]])
        out = np.asarray(bm.as_jax(utils.polynomial_features(X, degree=2, add_bias=False)))
        # P16-M2: 2 linear + 3 interaction -> 5 cols (previously 6 with a dead
        # leading allocation slot).
        assert out.shape == (1, 5)
        assert not np.any(np.all(out == 0, axis=0))


class TestNormalize:
    def test_normalize_unit_norm(self):
        X = bm.asarray([[3.0, 4.0]])
        out = np.asarray(bm.as_jax(utils.normalize(X)))
        # row normalized to unit l2 norm
        assert np.allclose(np.linalg.norm(out, axis=-1), 1.0, atol=1e-6)
        assert np.allclose(out, np.array([[0.6, 0.8]]), atol=1e-6)

    def test_normalize_zero_row_guarded(self):
        # zero-norm row must not produce NaN (the where(l2==0, 1, l2) guard)
        X = bm.asarray([[0.0, 0.0], [3.0, 4.0]])
        out = np.asarray(bm.as_jax(utils.normalize(X)))
        assert not np.any(np.isnan(out))
        assert np.allclose(out[0], 0.0)
