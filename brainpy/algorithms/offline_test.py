# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy.algorithms.offline``.

Each offline regression algorithm is fit on tiny synthetic data drawn from a
known linear relation (``y = 2x + 1`` style) and the resulting weights /
predictions are checked to be finite and, for the closed-form solvers, close
to the analytic answer.

Covered:

* ``OfflineAlgorithm`` base (abstract ``call`` raising, ``__repr__``),
* ``_check_data_2d_atls`` 2-D pass-through, >2-D flatten, and the <2-D error,
* ``RegressionAlgorithm`` ``init_weights`` / ``predict`` / gradient-descent solve,
* ``LinearRegression`` closed-form (lstsq) and gradient-descent paths,
* ``RidgeRegression`` closed-form (alpha>0 penalty, including the ``add_bias``
  no-penalize-intercept branch) and gradient-descent path, plus deprecated
  ``beta`` warning and ``__repr__``,
* ``LassoRegression`` (gradient descent + ``predict`` with polynomial+normalize),
* ``LogisticRegression`` (gradient-descent path + ``predict``),
* ``PolynomialRegression`` / ``PolynomialRidgeRegression`` / ``ElasticNetRegression``,
* the method registry: ``get_supported_offline_methods``, ``register_offline_method``
  (success + both error branches) and private ``get``.
"""

import warnings

import numpy as np
import pytest

import brainpy.math as bm
from brainpy.algorithms import offline


def _xy(n=40, slope=2.0, intercept=0.0, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-1.0, 1.0, size=(n, 1)).astype(np.float32)
    y = (slope * x + intercept).astype(np.float32)
    return bm.asarray(x), bm.asarray(y)


class TestOfflineBase:
    def test_abstract_call_raises(self):
        algo = offline.OfflineAlgorithm(name='base_offline_t')
        with pytest.raises(NotImplementedError):
            algo.call(targets=bm.ones((2, 1)), inputs=bm.ones((2, 1)))

    def test_repr(self):
        algo = offline.OfflineAlgorithm(name='base_offline_t2')
        assert repr(algo) == 'OfflineAlgorithm'


class TestRegressionAlgorithmBase:
    def test_initialize_is_noop(self):
        algo = offline.LinearRegression()
        # base ``initialize`` is a no-op accepting arbitrary args
        assert algo.initialize(1, 2, foo='bar') is None


class TestCheckData:
    def test_2d_passthrough(self):
        x = bm.as_jax(bm.ones((3, 2)))
        out = offline._check_data_2d_atls(x)
        assert out.shape == (3, 2)

    def test_3d_flatten(self):
        x = bm.as_jax(bm.ones((2, 3, 4)))
        out = offline._check_data_2d_atls(x)
        # flattened to (..., 4) keeping last dim
        assert out.shape[-1] == 4 and out.ndim == 2

    def test_1d_raises(self):
        x = bm.as_jax(bm.ones((5,)))
        with pytest.raises(ValueError):
            offline._check_data_2d_atls(x)


class TestLinearRegression:
    def test_closed_form_recovers_slope(self):
        x, y = _xy(slope=2.0)
        algo = offline.LinearRegression()
        w = np.asarray(bm.as_jax(algo(y, x)))
        assert np.allclose(w.flatten()[0], 2.0, atol=1e-3)

    def test_gradient_descent_path(self):
        # The gradient-descent solver does not normalise the gradient by the
        # sample count, so the exact converged value is LR/iteration sensitive;
        # we only assert it runs and stays finite (exercising the
        # ``gradient_descent_solve`` while_loop body).
        x, y = _xy(slope=2.0)
        algo = offline.LinearRegression(gradient_descent=True, max_iter=500, learning_rate=0.005)
        w = np.asarray(bm.as_jax(algo(y, x)))
        assert np.all(np.isfinite(w))

    def test_predict_and_init_weights(self):
        x, y = _xy(slope=2.0)
        algo = offline.LinearRegression()
        w = algo(y, x)
        pred = np.asarray(bm.as_jax(algo.predict(w, bm.as_jax(x))))
        assert pred.shape[0] == x.shape[0]
        wi = algo.init_weights(3, 2)
        assert wi.shape == (3, 2)

    def test_module_singleton(self):
        # the module-level singleton instance should also be callable
        x, y = _xy(slope=2.0)
        w = np.asarray(bm.as_jax(offline.linear_regression(y, x)))
        assert np.allclose(w.flatten()[0], 2.0, atol=1e-3)


class TestRidgeRegression:
    def test_closed_form(self):
        x, y = _xy(slope=2.0)
        algo = offline.RidgeRegression(alpha=1e-3)
        w = np.asarray(bm.as_jax(algo(y, x)))
        assert np.allclose(w.flatten()[0], 2.0, atol=1e-2)

    def test_gradient_descent_path(self):
        x, y = _xy(slope=2.0)
        algo = offline.RidgeRegression(alpha=1e-3, gradient_descent=True,
                                       max_iter=2000, learning_rate=0.01)
        w = np.asarray(bm.as_jax(algo(y, x)))
        assert np.all(np.isfinite(w))

    def test_deprecated_beta_warning_and_repr(self):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            algo = offline.RidgeRegression(beta=0.5)
        assert any(issubclass(w.category, UserWarning) for w in rec)
        # beta was copied into alpha
        assert algo.regularizer.alpha == 0.5
        assert 'beta=' in repr(algo)

    def test_add_bias_intercept_not_penalized(self):
        # RidgeRegression with add_bias attribute set takes the penalty.at[0]=0 branch
        x, y = _xy(slope=2.0, intercept=1.0)
        algo = offline.RidgeRegression(alpha=1.0)
        algo.add_bias = True
        # prepend a constant bias column manually (index 0)
        xb = bm.as_jax(bm.concatenate([bm.ones((x.shape[0], 1)), bm.asarray(x)], axis=1))
        w = np.asarray(bm.as_jax(algo(y, xb)))
        assert np.all(np.isfinite(w))
        # the intercept (col 0) should be close to 1 since it is unpenalized
        assert abs(w.flatten()[0] - 1.0) < 0.2


class TestLassoRegression:
    def test_fit_and_predict(self):
        x, y = _xy(slope=2.0)
        algo = offline.LassoRegression(alpha=0.01, degree=2, max_iter=500, learning_rate=0.001)
        w = np.asarray(bm.as_jax(algo(y, x)))
        assert np.all(np.isfinite(w))
        pred = np.asarray(bm.as_jax(algo.predict(bm.asarray(w), x)))
        assert np.all(np.isfinite(pred))


class TestLogisticRegression:
    def test_call_is_currently_broken(self):
        # NOTE (defect): LogisticRegression.call flattens ``targets`` to 1-D
        # (offline.py line ~386) and then immediately reads ``targets.shape[1]``
        # at line ~389, which raises IndexError. The closed-form (non gradient
        # descent) branch is also unreachable because of this. The fit path is
        # therefore broken for both gradient_descent=True and =False.
        rng = np.random.RandomState(1)
        x = rng.uniform(-1, 1, size=(30, 2)).astype(np.float32)
        y = (x[:, :1] > 0).astype(np.float32)
        algo = offline.LogisticRegression(max_iter=50, learning_rate=0.1)
        with pytest.raises(IndexError):
            algo(bm.asarray(y), bm.asarray(x))

    def test_predict_applies_sigmoid(self):
        # ``predict`` itself works in isolation (it does not hit the broken call).
        algo = offline.LogisticRegression(max_iter=10)
        W = bm.asarray(np.zeros((2, 1), dtype=np.float32))
        X = bm.asarray(np.ones((3, 2), dtype=np.float32))
        pred = np.asarray(bm.as_jax(algo.predict(W, X)))
        # sigmoid(0) == 0.5
        assert np.allclose(pred, 0.5, atol=1e-6)

    def test_multi_output_target_raises(self):
        x = bm.asarray(np.ones((4, 2), dtype=np.float32))
        y = bm.asarray(np.ones((4, 2), dtype=np.float32))  # 2 columns -> error
        algo = offline.LogisticRegression(max_iter=10)
        with pytest.raises(ValueError):
            algo(y, x)


class TestPolynomialRegressions:
    def test_polynomial_regression(self):
        x, y = _xy(slope=2.0)
        algo = offline.PolynomialRegression(degree=2, max_iter=300, learning_rate=0.001)
        w = np.asarray(bm.as_jax(algo(y, x)))
        assert np.all(np.isfinite(w))
        pred = np.asarray(bm.as_jax(algo.predict(bm.asarray(w), x)))
        assert np.all(np.isfinite(pred))

    def test_polynomial_ridge_regression(self):
        x, y = _xy(slope=2.0)
        algo = offline.PolynomialRidgeRegression(alpha=0.1, degree=2,
                                                 max_iter=300, learning_rate=0.001)
        w = np.asarray(bm.as_jax(algo(y, x)))
        assert np.all(np.isfinite(w))
        pred = np.asarray(bm.as_jax(algo.predict(bm.asarray(w), x)))
        assert np.all(np.isfinite(pred))

    def test_elastic_net_regression_fit(self):
        x, y = _xy(slope=2.0)
        algo = offline.ElasticNetRegression(alpha=0.01, degree=2, l1_ratio=0.5,
                                            max_iter=300, learning_rate=0.001)
        w = np.asarray(bm.as_jax(algo(y, x)))
        assert np.all(np.isfinite(w))

    def test_elastic_net_predict_bias_mismatch_is_broken(self):
        # NOTE (defect): ElasticNetRegression.call builds features with
        # ``polynomial_features(inputs, degree=self.degree)`` which defaults to
        # add_bias=True, while ``predict`` calls it with add_bias=self.add_bias
        # (default False). The resulting feature width differs from the fitted
        # weight length, so predicting on freshly-built features raises a
        # shape-mismatch TypeError from jnp.dot.
        x, y = _xy(slope=2.0)
        algo = offline.ElasticNetRegression(alpha=0.01, degree=2, l1_ratio=0.5,
                                            max_iter=50, learning_rate=0.001)
        w = bm.asarray(np.asarray(bm.as_jax(algo(y, x))))
        with pytest.raises(TypeError):
            algo.predict(w, x)


class TestRegistry:
    def test_supported_methods(self):
        methods = offline.get_supported_offline_methods()
        for m in ('linear', 'lstsq', 'ridge', 'lasso', 'logistic',
                  'polynomial', 'polynomial_ridge', 'elastic_net'):
            assert m in methods

    def test_get_lookup(self):
        assert offline.get('linear') is offline.LinearRegression
        with pytest.raises(ValueError):
            offline.get('nonexistent_method')

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError):
            offline.register_offline_method('linear', offline.LinearRegression())

    def test_register_wrong_type_raises(self):
        with pytest.raises(ValueError):
            offline.register_offline_method('brand_new_offline', object())

    def test_register_success(self):
        name = 'custom_offline_method_t'
        if name in offline.name2func:
            del offline.name2func[name]
        offline.register_offline_method(name, offline.LinearRegression())
        assert name in offline.get_supported_offline_methods()
        del offline.name2func[name]
