# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy.algorithms.online``.

Exercises the online training algorithms and the method registry:

* ``OnlineAlgorithm`` base (abstract ``call`` raising, ``register_target``
  no-op, ``__repr__``),
* ``RLS`` recursive least squares: target registration, scalar (1-D) and
  batched (2-D) updates, and convergence of repeated updates towards the
  true weights on a tiny ``y = W x`` problem,
* ``LMS`` least-mean-squares update (1-D and 2-D),
* ``get_supported_online_methods`` / ``register_online_method`` (success +
  both error branches) and the private ``get`` lookup.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import brainpy.math as bm
from brainpy.algorithms import online


class TestOnlineBase:
    def test_abstract_call_raises(self):
        algo = online.OnlineAlgorithm(name='base_online_t')
        with pytest.raises(NotImplementedError):
            algo.call(target=jnp.ones((1, 1)), input=jnp.ones((1, 1)), output=jnp.ones((1, 1)))

    def test_register_target_noop_and_repr(self):
        algo = online.OnlineAlgorithm(name='base_online_t2')
        # base register_target is a no-op (returns None, no error)
        assert algo.register_target(3) is None
        assert algo.__repr__() == 'OnlineAlgorithm'


class TestRLS:
    def test_register_target_creates_P(self):
        algo = online.RLS(alpha=0.5)
        algo.register_target(feature_in=3, identifier='node1')
        P = algo.implicit_vars['node1' + online.RLS.postfix]
        # P initialized to alpha * I
        assert np.allclose(np.asarray(bm.as_jax(P.value)), 0.5 * np.eye(3))

    def test_scalar_1d_update_runs(self):
        algo = online.RLS(alpha=1.0)
        algo.register_target(feature_in=2)
        # 1-D inputs get expanded to 2-D internally
        dw = algo(jnp.asarray([1.0]), jnp.asarray([1.0, 2.0]), jnp.asarray([0.0]))
        dw = np.asarray(bm.as_jax(dw))
        assert dw.shape == (2, 1)
        assert np.all(np.isfinite(dw))

    def test_batched_2d_update_runs(self):
        algo = online.RLS(alpha=1.0)
        algo.register_target(feature_in=2)
        target = jnp.asarray([[1.0], [2.0]])
        inp = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
        out = jnp.asarray([[0.0], [0.0]])
        dw = np.asarray(bm.as_jax(algo(target, inp, out)))
        assert dw.shape == (2, 1)
        assert np.all(np.isfinite(dw))

    def test_repeated_updates_converge(self):
        # Train a 2-input -> 1-output linear map y = W x via RLS and check the
        # accumulated weight estimate approaches the true weights.
        rng = np.random.RandomState(0)
        W_true = np.array([[2.0], [-1.0]])
        algo = online.RLS(alpha=1.0)
        algo.register_target(feature_in=2)
        W = np.zeros((2, 1))
        for _ in range(200):
            x = rng.randn(1, 2)
            target = x @ W_true
            output = x @ W
            dw = np.asarray(bm.as_jax(algo(jnp.asarray(target), jnp.asarray(x), jnp.asarray(output))))
            W = W + dw
        assert np.allclose(W, W_true, atol=1e-2)


class TestLMS:
    def test_lms_1d_update(self):
        algo = online.LMS(alpha=0.1)
        dw = algo(jnp.asarray([1.0]), jnp.asarray([1.0, 2.0]), jnp.asarray([0.0]))
        dw = np.asarray(bm.as_jax(dw))
        assert dw.shape == (2, 1)
        assert np.all(np.isfinite(dw))

    def test_lms_2d_update_sign(self):
        algo = online.LMS(alpha=0.5)
        # error = output - target = -1 ; dw = -alpha * outer(input, error)
        target = jnp.asarray([[1.0]])
        inp = jnp.asarray([[2.0, 4.0]])
        output = jnp.asarray([[0.0]])
        dw = np.asarray(bm.as_jax(algo(target, inp, output)))
        # -alpha * input * error = -0.5 * [2,4] * (-1) = [1, 2]
        assert np.allclose(dw.flatten(), np.array([1.0, 2.0]), atol=1e-6)


class TestRegistry:
    def test_supported_methods(self):
        methods = online.get_supported_online_methods()
        assert 'rls' in methods and 'lms' in methods

    def test_get_lookup(self):
        assert online.get('rls') is online.RLS
        with pytest.raises(ValueError):
            online.get('does_not_exist')

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError):
            online.register_online_method('rls', online.RLS(alpha=0.1))

    def test_register_wrong_type_raises(self):
        with pytest.raises(ValueError):
            online.register_online_method('brand_new_online', object())

    def test_register_success(self):
        name = 'custom_online_method_t'
        if name in online.name2func:
            del online.name2func[name]
        online.register_online_method(name, online.LMS(alpha=0.2))
        assert name in online.get_supported_online_methods()
        # cleanup to avoid polluting global registry across reruns
        del online.name2func[name]
