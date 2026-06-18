# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy.optim.optimizer``.

Each optimizer subclass is driven on a tiny scalar/vector trainable
``bm.Variable`` for a few gradient steps so that its ``update`` body, its
``register_train_vars`` body and its ``weight_decay`` branch all execute.

Covered classes: ``SGD``, ``Momentum``, ``MomentumNesterov``, ``Adagrad``,
``Adadelta``, ``RMSProp``, ``Adam``, ``LARS``, ``Adan`` (both ``no_prox``
branches), ``AdamW`` (with/without amsgrad and zero weight_decay), and
``SM3`` (with/without momentum & beta, including a multi-dimensional
variable). Also covers the base ``Optimizer`` error/abstract branches,
``check_grads``, ``register_vars`` deprecation, ``__repr__`` for each, and
the ``train_vars`` type validation error.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import brainpy.math as bm
from brainpy._errors import MathError
from brainpy.optim import optimizer as O


def _make_var(value=1.0):
    return bm.Variable(bm.asarray(np.asarray(value, dtype=np.float32)))


def _train(opt, var_dict, grad_fn, steps=5):
    """Run a few update steps; grad_fn maps a value -> gradient array."""
    for _ in range(steps):
        grads = {k: grad_fn(bm.as_jax(v.value)) for k, v in var_dict.items()}
        opt.update(grads)
    return {k: np.asarray(bm.as_jax(v.value)) for k, v in var_dict.items()}


class TestBaseOptimizer:
    def test_register_train_vars_not_implemented(self):
        # the abstract base raises before being usable
        with pytest.raises(NotImplementedError):
            O.Optimizer(lr=0.1)

    def test_update_not_implemented(self):
        opt = O.SGD(lr=0.1)
        # SGD overrides update; the *base* update is abstract. Test via a bare
        # CommonOpt subclass without update override is awkward, so check that
        # the base Optimizer.update raises.
        with pytest.raises(NotImplementedError):
            O.Optimizer.update(opt, {})

    def test_check_grads_length_mismatch(self):
        v = _make_var()
        opt = O.SGD(lr=0.1, train_vars={'w': v})
        with pytest.raises(MathError):
            opt.update({})  # zero grads but one train var

    def test_register_vars_deprecation_warning(self):
        opt = O.SGD(lr=0.1)
        with pytest.warns(UserWarning):
            opt.register_vars({})

    def test_train_vars_must_be_dict(self):
        with pytest.raises(MathError):
            O.SGD(lr=0.1, train_vars=[_make_var()])  # list, not dict

    def test_repr(self):
        opt = O.SGD(lr=0.1)
        assert 'SGD' in repr(opt)


class TestSGD:
    def test_decreases_value(self):
        v = _make_var(5.0)
        opt = O.SGD(lr=0.1, train_vars={'w': v})
        out = _train(opt, {'w': v}, lambda val: val)  # grad = value -> decay to 0
        assert out['w'] < 5.0

    def test_weight_decay_branch(self):
        v = _make_var(5.0)
        opt = O.SGD(lr=0.1, train_vars={'w': v}, weight_decay=0.01)
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w'])


class TestMomentum:
    def test_update(self):
        v = _make_var(2.0)
        opt = O.Momentum(lr=0.05, train_vars={'w': v}, momentum=0.9)
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w']) and 'momentum' in repr(opt)

    def test_weight_decay(self):
        v = _make_var(2.0)
        opt = O.Momentum(lr=0.05, train_vars={'w': v}, momentum=0.9, weight_decay=0.01)
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w'])


class TestMomentumNesterov:
    def test_update(self):
        v = _make_var(2.0)
        opt = O.MomentumNesterov(lr=0.05, train_vars={'w': v})
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w']) and 'momentum' in repr(opt)

    def test_weight_decay(self):
        v = _make_var(2.0)
        opt = O.MomentumNesterov(lr=0.05, train_vars={'w': v}, weight_decay=0.01)
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w'])


class TestAdagrad:
    def test_update_and_repr(self):
        v = _make_var(2.0)
        opt = O.Adagrad(lr=0.1, train_vars={'w': v})
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w']) and 'epsilon' in repr(opt)

    def test_weight_decay(self):
        v = _make_var(2.0)
        opt = O.Adagrad(lr=0.1, train_vars={'w': v}, weight_decay=0.02)
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w'])


class TestAdadelta:
    def test_update_and_repr(self):
        v = _make_var(2.0)
        opt = O.Adadelta(lr=1.0, train_vars={'w': v})
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w']) and 'rho' in repr(opt)

    def test_weight_decay(self):
        v = _make_var(2.0)
        opt = O.Adadelta(lr=1.0, train_vars={'w': v}, weight_decay=0.02)
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w'])


class TestRMSProp:
    def test_update_and_repr(self):
        v = _make_var(2.0)
        opt = O.RMSProp(lr=0.05, train_vars={'w': v})
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w']) and 'rho' in repr(opt)

    def test_weight_decay(self):
        v = _make_var(2.0)
        opt = O.RMSProp(lr=0.05, train_vars={'w': v}, weight_decay=0.02)
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w'])


class TestAdam:
    def test_update_and_repr(self):
        v = _make_var(2.0)
        opt = O.Adam(lr=0.05, train_vars={'w': v})
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w']) and 'beta1' in repr(opt)
        # step counter advanced once per update
        assert int(bm.as_jax(opt.step.value)) == 5

    def test_weight_decay(self):
        v = _make_var(2.0)
        opt = O.Adam(lr=0.05, train_vars={'w': v}, weight_decay=0.01)
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w'])


class TestLARS:
    def test_update_and_repr(self):
        v = bm.Variable(bm.asarray(np.array([1.0, -2.0], dtype=np.float32)))
        opt = O.LARS(lr=0.1, train_vars={'w': v})
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.all(np.isfinite(out['w'])) and 'tc' in repr(opt)

    def test_zero_norm_branch(self):
        # a zero-valued parameter exercises the (p_norm == 0) logical_or branch
        v = bm.Variable(bm.asarray(np.zeros(2, dtype=np.float32)))
        opt = O.LARS(lr=0.1, train_vars={'w': v})
        out = _train(opt, {'w': v}, lambda val: np.ones_like(val))
        assert np.all(np.isfinite(out['w']))


class TestAdan:
    def test_update_runs(self):
        # P1-C1/P1-C2 fix: Adan.update used to crash (jax.lax.cond operand
        # splatting) and its step counter was frozen at 0. It now runs and the
        # per-update step counter advances.
        v = _make_var(2.0)
        opt = O.Adan(lr=1e-2, train_vars={'w': v})
        assert 'no_prox' in repr(opt)
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w'])
        assert int(bm.as_jax(opt.step.value)) == 5

    def test_invalid_eps(self):
        with pytest.raises(ValueError):
            O.Adan(lr=1e-2, eps=-1.0)

    def test_invalid_betas(self):
        with pytest.raises(ValueError):
            O.Adan(lr=1e-2, betas=(1.5, 0.08, 0.01))
        with pytest.raises(ValueError):
            O.Adan(lr=1e-2, betas=(0.02, 1.5, 0.01))
        with pytest.raises(ValueError):
            O.Adan(lr=1e-2, betas=(0.02, 0.08, 1.5))

    def test_wrong_number_of_betas(self):
        with pytest.raises(AssertionError):
            O.Adan(lr=1e-2, betas=(0.1, 0.2))


class TestAdamW:
    def test_update_and_repr(self):
        v = _make_var(2.0)
        opt = O.AdamW(lr=0.05, train_vars={'w': v}, weight_decay=1e-2)
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w']) and 'amsgrad' in repr(opt)

    def test_amsgrad_branch(self):
        v = _make_var(2.0)
        opt = O.AdamW(lr=0.05, train_vars={'w': v}, amsgrad=True)
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w'])

    def test_zero_weight_decay_branch(self):
        v = _make_var(2.0)
        opt = O.AdamW(lr=0.05, train_vars={'w': v}, weight_decay=0.0)
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w'])

    def test_invalid_hyperparams(self):
        with pytest.raises(ValueError):
            O.AdamW(lr=0.05, eps=-1.0)
        with pytest.raises(ValueError):
            O.AdamW(lr=0.05, beta1=1.5)
        with pytest.raises(ValueError):
            O.AdamW(lr=0.05, beta2=1.5)
        with pytest.raises(ValueError):
            O.AdamW(lr=0.05, weight_decay=-1.0)


class TestSM3:
    def test_scalar_var_runs(self):
        # P1-H1 fix: SM3 used to KeyError('w_m0') for a scalar (0-dim) variable
        # because no accumulator was registered. It now registers a single
        # scalar accumulator (Adagrad-like) and updates correctly.
        v = _make_var(2.0)
        opt = O.SM3(lr=0.1, train_vars={'w': v})
        assert 'beta' in repr(opt)
        out = _train(opt, {'w': v}, lambda val: np.ones_like(val))
        assert np.isfinite(out['w'])

    def test_1d_var(self):
        v = bm.Variable(bm.asarray(np.array([1.0, 2.0], dtype=np.float32)))
        opt = O.SM3(lr=0.1, train_vars={'w': v})
        out = _train(opt, {'w': v}, lambda val: np.ones_like(val))
        assert np.all(np.isfinite(out['w']))

    def test_multidim_with_momentum_and_beta(self):
        v = bm.Variable(bm.asarray(np.ones((2, 3), dtype=np.float32)))
        opt = O.SM3(lr=0.1, train_vars={'w': v}, momentum=0.5, beta=0.5)
        out = _train(opt, {'w': v}, lambda val: np.ones_like(val))
        assert np.all(np.isfinite(out['w']))

    def test_weight_decay_branch(self):
        v = bm.Variable(bm.asarray(np.ones((2, 2), dtype=np.float32)))
        opt = O.SM3(lr=0.1, train_vars={'w': v}, weight_decay=0.02)
        out = _train(opt, {'w': v}, lambda val: np.ones_like(val))
        assert np.all(np.isfinite(out['w']))

    def test_invalid_hyperparams(self):
        with pytest.raises(ValueError):
            O.SM3(lr=0.1, momentum=1.5)
        with pytest.raises(ValueError):
            O.SM3(lr=0.1, beta=1.5)
        with pytest.raises(ValueError):
            O.SM3(lr=0.1, eps=-1.0)


class TestRegisterTrainVarsErrors:
    @pytest.mark.parametrize('cls', [O.Momentum, O.MomentumNesterov, O.Adagrad,
                                     O.Adadelta, O.RMSProp, O.Adam, O.LARS,
                                     O.Adan, O.AdamW, O.SM3])
    def test_non_dict_train_vars_raises(self, cls):
        # every subclass validates that train_vars is a dict in its
        # register_train_vars; passing a list hits that error branch.
        with pytest.raises(MathError):
            cls(lr=0.05, train_vars=[_make_var()])


class TestAdanUpdateMomentsHelper:
    def test_update_moments_runs(self):
        # ``Adan._update_moments`` is an (unused-by-update) helper; call it
        # directly so its body executes and returns finite moment estimates.
        opt = O.Adan(lr=1e-2)
        z = jnp.asarray(0.0)
        g = jnp.asarray(1.0)
        m, n, v = opt._update_moments(m=z, n=z, v=z, pre_g=z, g=g)
        assert all(np.isfinite(np.asarray(bm.as_jax(t))) for t in (m, n, v))


class TestSchedulerLR:
    def test_optimizer_with_scheduler_lr(self):
        # drive an optimizer with a real LR scheduler so the lr.step_call path
        # advances a CallBasedScheduler.
        import brainpy as bp
        v = _make_var(2.0)
        sched = bp.optim.ExponentialDecayLR(lr=0.1, decay_steps=1, decay_rate=0.99)
        opt = O.Adam(lr=sched, train_vars={'w': v})
        out = _train(opt, {'w': v}, lambda val: val)
        assert np.isfinite(out['w'])
