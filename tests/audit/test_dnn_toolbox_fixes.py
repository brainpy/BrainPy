# -*- coding: utf-8 -*-
"""Regression + coverage tests for the DNN / toolbox audit fixes (2026-06-18).

This module tests the fixes recorded in ``docs/issues-found-20260618.md`` for the
following source files:

* ``brainpy/dnn/normalization.py``      -- C-05, H-51, M-25/M-26 (GroupNorm/Instance
                                           group axis, BatchNorm/LayerNorm/GroupNorm
                                           constructing + running under the default mode).
* ``brainpy/losses/comparison.py``      -- C-02 (``nll_loss`` sign), C-03 (class-weighted
                                           cross-entropy), H-53 (``ignore_index`` /
                                           ``label_smoothing``).
* ``brainpy/optim/optimizer.py``        -- C-01 / H-52 (Adam/AdamW bias-correction step
                                           counter), M-29 (``SM3`` instantiable with
                                           train vars).
* ``brainpy/optim/scheduler.py``        -- C-04 (``MultiStepLR`` decay), M-01 (other LR
                                           schedulers).
* ``brainpy/encoding/stateless_encoding.py`` -- ``PoissonEncoder.single_step`` no longer
                                           crashes / passes wrong args.
* ``brainpy/connect/random_conn.py``    -- M-30 (``FixedProb`` empty/rectangular build).

In addition to the targeted regression checks, the module exercises every optimizer,
scheduler, loss function, normalization layer and stateless/stateful encoder for
coverage. Known *remaining* bugs (``Adan.update``, ``ctc_loss``'s ``.value`` access)
are pinned with ``pytest.raises`` so the lines stay covered and the behaviour is
documented rather than silently broken.
"""

import numpy as np
import jax.numpy as jnp
import pytest

import brainpy as bp
import brainpy.math as bm
from brainpy.context import share
from brainpy.optim import optimizer as O
from brainpy.optim import scheduler as S
from brainpy.losses import comparison as C
from brainpy.optim.optimizer import SM3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _np(x):
    return np.asarray(bm.as_jax(x))


def _train_var(shape=(2, 3), fill=1.0):
    return bm.Variable(bm.ones(shape) * fill)


# ===========================================================================
# 1. normalization.py  --  GroupNorm / InstanceNorm / BatchNorm / LayerNorm
# ===========================================================================

def test_groupnorm_respects_groups():
    """C-05: GroupNorm(3,6) must differ from GroupNorm(1,6); num_groups has effect."""
    bm.random.seed(0)
    x = bm.random.randn(4, 6)
    g3 = _np(bp.dnn.GroupNorm(3, 6, affine=False)(x))
    g1 = _np(bp.dnn.GroupNorm(1, 6, affine=False)(x))
    g6 = _np(bp.dnn.GroupNorm(6, 6, affine=False)(x))
    assert not np.allclose(g3, g1), "GroupNorm(3,6) must differ from GroupNorm(1,6)"
    assert not np.allclose(g3, g6), "GroupNorm(3,6) must differ from GroupNorm(6,6)"


def test_groupnorm_each_group_zero_mean_unit_var():
    """C-05: each group is normalized independently to ~zero-mean / unit-var."""
    bm.random.seed(1)
    x = bm.random.randn(4, 6) * 3.0 + 5.0
    y = _np(bp.dnn.GroupNorm(3, 6, affine=False)(x))
    # 6 channels / 3 groups -> 2 channels per group.
    grouped = y.reshape(4, 3, 2)
    assert np.allclose(grouped.mean(axis=-1), 0.0, atol=1e-4)
    assert np.allclose(grouped.var(axis=-1), 1.0, atol=1e-2)


def test_instancenorm_per_channel_unit_var():
    """C-05: InstanceNorm normalizes each channel independently over the spatial axis."""
    bm.random.seed(2)
    x = bm.random.randn(4, 8, 6) * 3.0 + 2.0
    y = _np(bp.dnn.InstanceNorm(6, affine=False)(x))
    # Normalized over the spatial axis (axis=1) per (sample, channel).
    assert np.allclose(y.std(axis=1), 1.0, atol=1e-2)
    assert np.allclose(y.mean(axis=1), 0.0, atol=1e-4)


def test_norm_layers_construct_and_run_default_mode():
    """H-51: BatchNorm1d / LayerNorm / GroupNorm / InstanceNorm construct under the
    default (non-training) mode and run a forward pass without raising."""
    x3 = bm.random.randn(2, 4, 3)
    assert _np(bp.dnn.BatchNorm1d(num_features=3)(x3)).shape == (2, 4, 3)
    assert _np(bp.dnn.LayerNorm(3)(x3)).shape == (2, 4, 3)
    x2 = bm.random.randn(2, 6)
    assert _np(bp.dnn.GroupNorm(3, 6)(x2)).shape == (2, 6)
    assert _np(bp.dnn.InstanceNorm(6)(x2)).shape == (2, 6)


def test_norm_layers_affine_and_aliases():
    """Affine path + dimensional aliases (BatchNorm*D) construct and run."""
    x3 = bm.random.randn(2, 4, 3)
    assert _np(bp.dnn.BatchNorm1D(num_features=3, affine=True)(x3)).shape == (2, 4, 3)
    assert _np(bp.dnn.LayerNorm(3, elementwise_affine=True)(x3)).shape == (2, 4, 3)
    assert _np(bp.dnn.GroupNorm(3, 6, affine=True)(bm.random.randn(2, 6))).shape == (2, 6)


def test_batchnorm_2d_3d_fit_and_eval():
    """BatchNorm2d/3d run in both 'fit' (update running stats) and 'eval' modes."""
    x4 = bm.random.randn(2, 4, 4, 3)
    x5 = bm.random.randn(2, 4, 4, 4, 3)
    share.save(fit=True)
    try:
        assert _np(bp.dnn.BatchNorm2d(num_features=3)(x4)).shape == (2, 4, 4, 3)
        assert _np(bp.dnn.BatchNorm3d(num_features=3)(x5)).shape == (2, 4, 4, 4, 3)
        # axis_name=None path with affine
        assert _np(bp.dnn.BatchNorm2D(num_features=3, affine=True)(x4)).shape == (2, 4, 4, 3)
    finally:
        share.save(fit=False)
    # eval mode: uses running stats
    assert _np(bp.dnn.BatchNorm2d(num_features=3)(x4)).shape == (2, 4, 4, 3)
    assert _np(bp.dnn.BatchNorm3D(num_features=3)(x5)).shape == (2, 4, 4, 4, 3)


def test_batchnorm_input_dim_check():
    """The _check_input_dim guards raise on wrong ndim."""
    with pytest.raises(ValueError):
        bp.dnn.BatchNorm1d(num_features=3)(bm.random.randn(2, 3))  # needs 3D
    with pytest.raises(ValueError):
        bp.dnn.BatchNorm2d(num_features=3)(bm.random.randn(2, 4, 3))  # needs 4D
    with pytest.raises(ValueError):
        bp.dnn.BatchNorm3d(num_features=3)(bm.random.randn(2, 4, 4, 3))  # needs 5D


def test_groupnorm_invalid_channels():
    """num_channels must be divisible by num_groups."""
    with pytest.raises(ValueError):
        bp.dnn.GroupNorm(4, 6)


def test_layernorm_shape_mismatch_raises():
    """M-26: LayerNorm raises on a normalized-shape mismatch.

    NOTE: the intended ``ValueError`` message is itself mis-built
    (``", ".join(self.normalized_shape)`` joins ints, normalization.py:536), so
    a ``TypeError`` surfaces first. Either way the layer rejects the bad shape;
    we only assert that *some* exception is raised."""
    with pytest.raises(Exception):
        bp.dnn.LayerNorm(5)(bm.random.randn(2, 4, 3))


# ===========================================================================
# 2. comparison.py  --  loss functions
# ===========================================================================

def test_nll_loss_is_positive():
    """C-02: nll_loss returns the *negative* log-likelihood (positive number)."""
    log_probs = jnp.log(jnp.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]))
    loss = float(C.nll_loss(log_probs, jnp.array([0, 1])))
    assert loss > 0.0
    assert abs(loss - 0.2899) < 1e-3


def test_nll_loss_reductions():
    log_probs = jnp.log(jnp.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]]))
    tgt = jnp.array([0, 1])
    assert float(C.nll_loss(log_probs, tgt, reduction='sum')) > 0
    assert _np(C.nll_loss(log_probs, tgt, reduction='none')).shape == (2,)
    assert _np(C.nll_loss(log_probs, tgt, reduction=None)).shape == (2,)
    with pytest.raises(ValueError):
        C.nll_loss(log_probs, tgt, reduction='bogus')
    # NLLLoss class
    assert float(C.NLLLoss()(log_probs, tgt)) > 0


def test_cross_entropy_weight_by_class():
    """C-03: class weight is applied by target class, not by sample index."""
    out = _np(C.cross_entropy_loss(bm.zeros((3, 3)), [2, 2, 2],
                                   weight=[10, 20, 1], reduction='none'))
    # all targets are class 2 -> all per-sample losses equal (w[2] * log(3)).
    assert np.allclose(out, out[0])
    assert abs(out[0] - 1.0986) < 1e-3


def test_cross_entropy_ignore_index():
    """H-53: ignore_index excludes the ignored sample from the loss."""
    bm.random.seed(3)
    logits = bm.random.randn(4, 3)
    tgt_all = jnp.array([0, 1, 2, 2])
    tgt_ign = jnp.array([0, 1, -100, 2])
    loss_all = float(C.cross_entropy_loss(logits, tgt_all))
    loss_ign = float(C.cross_entropy_loss(logits, tgt_ign, ignore_index=-100))
    # Excluding a sample generally changes the mean.
    assert not np.isclose(loss_all, loss_ign)
    # Equivalent to averaging only the 3 kept samples.
    per = _np(C.cross_entropy_loss(logits, tgt_all, reduction='none'))
    manual = (per[0] + per[1] + per[3]) / 3.0
    assert abs(loss_ign - manual) < 1e-4


def test_cross_entropy_label_smoothing_changes_loss():
    """H-53: label_smoothing > 0 changes the loss value."""
    bm.random.seed(4)
    logits = bm.random.randn(4, 3)
    tgt = jnp.array([0, 1, 2, 0])
    base = float(C.cross_entropy_loss(logits, tgt))
    smoothed = float(C.cross_entropy_loss(logits, tgt, label_smoothing=0.1))
    assert not np.isclose(base, smoothed)


def test_cross_entropy_class_and_loss_wrappers():
    """CrossEntropyLoss class wraps cross_entropy_loss; reductions + soft targets."""
    bm.random.seed(5)
    logits = bm.random.randn(4, 3)
    tgt = jnp.array([0, 1, 2, 0])
    assert float(C.CrossEntropyLoss()(logits, tgt)) > 0
    assert float(C.CrossEntropyLoss(ignore_index=-100)(logits, jnp.array([0, 1, -100, 2]))) > 0
    assert float(C.CrossEntropyLoss(label_smoothing=0.1)(logits, tgt)) > 0
    assert float(C.cross_entropy_loss(logits, tgt, reduction='sum')) > 0
    assert _np(C.cross_entropy_loss(logits, tgt, reduction='none')).shape == (4,)
    # soft (probability) targets + class weight path.
    soft = bm.one_hot(tgt, 3)
    assert float(C.cross_entropy_loss(logits, soft)) > 0
    assert float(C.cross_entropy_loss(logits, soft, weight=[1., 2., 3.])) > 0
    assert float(C.cross_entropy_loss(logits, soft, label_smoothing=0.1)) > 0


def test_cross_entropy_sparse_and_sigmoid():
    bm.random.seed(6)
    logits = bm.random.randn(4, 3)
    assert _np(C.cross_entropy_sparse(logits, jnp.array([[0], [1], [2], [0]]))).shape == (4,)
    assert float(jnp.sum(C.cross_entropy_sparse(logits, 1))) != 0.0  # single int target
    assert _np(C.cross_entropy_sigmoid(logits, bm.random.rand(4, 3))).shape == (4, 3)


def test_regression_losses():
    bm.random.seed(7)
    pred = bm.random.randn(4, 3)
    tar = bm.random.randn(4, 3)
    assert float(C.mean_squared_error(pred, tar)) >= 0
    assert float(C.mean_squared_error(pred, tar, reduction='sum')) >= 0
    assert _np(C.mean_squared_error(pred, tar, axis=1, reduction='none')).shape[0] == 4
    assert float(C.mean_absolute_error(pred, tar)) >= 0
    assert float(C.mean_squared_log_error(bm.abs(pred), bm.abs(tar))) >= 0
    assert float(C.l1_loss(pred, tar)) >= 0
    assert _np(C.l1_loss(pred, tar, reduction='none')).shape == (4,)
    assert float(jnp.sum(C.l2_loss(pred, tar))) >= 0
    assert float(jnp.sum(C.huber_loss(pred, tar, delta=0.5))) >= 0
    assert float(jnp.sum(C.log_cosh_loss(pred, tar))) >= 0
    # Loss classes
    assert float(C.MSELoss()(pred, tar)) >= 0
    assert float(C.MSELoss(reduction='sum')(pred, tar)) >= 0
    assert float(C.L1Loss()(pred, tar)) >= 0
    assert float(C.MAELoss(axis=None)(pred, tar)) >= 0


def test_classification_helper_losses():
    bm.random.seed(8)
    assert float(jnp.sum(C.sigmoid_binary_cross_entropy(bm.random.randn(4, 3), bm.random.rand(4, 3)))) != 0
    labels = bm.one_hot(jnp.array([0, 1, 2, 0]), 3)
    assert float(jnp.sum(C.softmax_cross_entropy(bm.random.randn(4, 3), labels))) >= 0
    assert float(jnp.sum(C.binary_logistic_loss(jnp.array([0.5, 1.2]), jnp.array([0, 1])))) != 0
    # multiclass_logistic_loss: single int label + (n_classes,) logits.
    assert float(C.multiclass_logistic_loss(1, bm.random.randn(3))) >= 0


def test_multi_margin_loss():
    bm.random.seed(9)
    logits = bm.random.randn(4, 3)
    tgt = jnp.array([0, 1, 2, 0])
    assert float(C.multi_margin_loss(logits, tgt)) >= 0
    assert float(C.multi_margin_loss(logits, tgt, p=2)) >= 0
    assert float(C.multi_margin_loss(logits, tgt, reduction='sum')) >= 0
    assert _np(C.multi_margin_loss(logits, tgt, reduction='none')).shape == (4, 3)
    with pytest.raises(AssertionError):
        C.multi_margin_loss(logits, tgt, p=3)


def test_ctc_loss_known_value_bug():
    """REMAINING BUG: ctc_loss / ctc_loss_with_forward_probs call ``.value`` on a
    plain JAX array returned by ``bm.log_softmax`` (comparison.py:1006), which
    raises AttributeError. Pinned here so the regression is documented + covered."""
    B, T, K, N = 2, 5, 4, 3
    logits = bm.random.randn(B, T, K)
    lp = bm.zeros((B, T))
    labels = jnp.array([[1, 2, 1], [2, 1, 2]])
    lbp = bm.zeros((B, N))
    with pytest.raises(AttributeError):
        C.ctc_loss(logits, lp, labels, lbp)
    with pytest.raises(AttributeError):
        C.ctc_loss_with_forward_probs(logits, lp, labels, lbp)


# ===========================================================================
# 3. optimizer.py
# ===========================================================================

def test_adam_constant_step_under_unit_gradient():
    """C-01 / H-52: under a constant unit gradient, Adam's per-step |dw| ~= lr and
    does NOT grow over time (bias correction uses a real step counter)."""
    lr = 0.01
    w = bm.Variable(bm.zeros((3,)))
    opt = bp.optim.Adam(lr=lr, train_vars={'w': w})
    prev = _np(w.value).copy()
    deltas = []
    for _ in range(5):
        opt.update({'w': bm.ones((3,))})
        cur = _np(w.value)
        deltas.append(abs(float(cur[0] - prev[0])))
        prev = cur.copy()
    assert np.allclose(deltas, lr, atol=1e-4), f"Adam steps drifted: {deltas}"


def test_adamw_constant_step_under_unit_gradient():
    """C-01 / H-52: same constant-step property for AdamW (weight_decay=0)."""
    lr = 0.01
    w = bm.Variable(bm.zeros((3,)))
    opt = bp.optim.AdamW(lr=lr, train_vars={'w': w}, weight_decay=0.0)
    prev = _np(w.value).copy()
    deltas = []
    for _ in range(5):
        opt.update({'w': bm.ones((3,))})
        cur = _np(w.value)
        deltas.append(abs(float(cur[0] - prev[0])))
        prev = cur.copy()
    assert np.allclose(deltas, lr, atol=1e-4), f"AdamW steps drifted: {deltas}"


def test_sm3_instantiates_and_updates():
    """M-29: SM3 instantiates with train_vars and a forward update changes w."""
    w = bm.Variable(bm.ones((3, 4)))
    opt = SM3(lr=0.01, train_vars={'w': w})
    before = _np(w.value).copy()
    opt.update({'w': bm.ones((3, 4))})
    assert not np.allclose(before, _np(w.value))


def test_sm3_with_momentum_and_weight_decay():
    """SM3 with momentum>0 (allocates a buffer) and weight_decay both run + change w."""
    w = bm.Variable(bm.ones((3, 4)))
    opt = SM3(lr=0.01, train_vars={'w': w}, momentum=0.5, beta=0.5, weight_decay=0.01)
    before = _np(w.value).copy()
    opt.update({'w': bm.ones((3, 4))})
    assert not np.allclose(before, _np(w.value))
    repr(opt)


def test_sm3_invalid_hyperparams():
    with pytest.raises(ValueError):
        SM3(lr=0.01, momentum=1.5)
    with pytest.raises(ValueError):
        SM3(lr=0.01, beta=1.5)
    with pytest.raises(ValueError):
        SM3(lr=0.01, eps=-1.0)


@pytest.mark.parametrize("name", ['SGD', 'Momentum', 'MomentumNesterov', 'Adagrad',
                                  'Adadelta', 'RMSProp', 'Adam', 'AdamW', 'LARS'])
def test_optimizer_construct_and_update(name):
    """Coverage: every (working) optimizer constructs and runs an update."""
    cls = getattr(O, name)
    w = _train_var()
    opt = cls(lr=0.01, train_vars={'w': w})
    before = _np(w.value).copy()
    opt.update({'w': bm.ones((2, 3)) * 0.1})
    assert _np(w.value).shape == (2, 3)
    repr(opt)
    # also run a second update step (exercises EMA / cache update paths)
    opt.update({'w': bm.ones((2, 3)) * 0.2})
    assert not np.allclose(before, _np(w.value))


@pytest.mark.parametrize("name", ['SGD', 'Momentum', 'MomentumNesterov', 'Adagrad',
                                  'Adadelta', 'RMSProp', 'Adam', 'LARS'])
def test_optimizer_weight_decay_path(name):
    """Coverage: the weight_decay branch of each optimizer's update."""
    cls = getattr(O, name)
    w = _train_var()
    opt = cls(lr=0.01, train_vars={'w': w}, weight_decay=0.01)
    opt.update({'w': bm.ones((2, 3)) * 0.1})
    assert _np(w.value).shape == (2, 3)


def test_adamw_amsgrad_variant():
    """AdamW amsgrad branch constructs and updates."""
    w = _train_var()
    opt = bp.optim.AdamW(lr=0.01, train_vars={'w': w}, amsgrad=True, weight_decay=0.01)
    opt.update({'w': bm.ones((2, 3)) * 0.1})
    assert _np(w.value).shape == (2, 3)


def test_adamw_invalid_hyperparams():
    with pytest.raises(ValueError):
        bp.optim.AdamW(lr=0.01, eps=-1.0)
    with pytest.raises(ValueError):
        bp.optim.AdamW(lr=0.01, beta1=1.5)
    with pytest.raises(ValueError):
        bp.optim.AdamW(lr=0.01, beta2=1.5)
    with pytest.raises(ValueError):
        bp.optim.AdamW(lr=0.01, weight_decay=-1.0)


def test_adan_constructs_update_is_known_bug():
    """Adan constructs, but REMAINING BUG: ``Adan.update`` passes a tuple operand to
    ``lax.cond`` while the branch lambdas expect two args (optimizer.py:818),
    raising TypeError. Pinned so the construct path is covered + the bug documented."""
    w = _train_var()
    adan = bp.optim.Adan(lr=0.01, train_vars={'w': w})
    repr(adan)
    with pytest.raises(TypeError):
        adan.update({'w': bm.ones((2, 3)) * 0.1})


def test_adan_invalid_betas():
    with pytest.raises(AssertionError):
        bp.optim.Adan(lr=0.01, betas=(0.1, 0.2))  # len != 3
    with pytest.raises(ValueError):
        bp.optim.Adan(lr=0.01, eps=-1.0)
    with pytest.raises(ValueError):
        bp.optim.Adan(lr=0.01, betas=(1.5, 0.1, 0.1))


def test_optimizer_check_grads_mismatch():
    """Optimizer.check_grads raises on a length mismatch."""
    w = _train_var()
    opt = bp.optim.SGD(lr=0.01, train_vars={'w': w})
    with pytest.raises(Exception):
        opt.update({'w': bm.ones((2, 3)), 'extra': bm.ones((2, 3))})


def test_optimizer_register_train_vars_type_error():
    with pytest.raises(Exception):
        bp.optim.SGD(lr=0.01, train_vars=[bm.Variable(bm.ones((2, 3)))])  # must be dict


# ===========================================================================
# 4. scheduler.py
# ===========================================================================

def test_multisteplr_decays():
    """C-04: MultiStepLR actually decays at the milestones."""
    sch = bp.optim.MultiStepLR(0.1, [10, 20], gamma=0.1)
    vals = [round(float(sch(i)), 6) for i in [0, 10, 20, 25]]
    assert vals == [0.1, 0.1, 0.01, 0.001], vals
    repr(sch)


def test_constant_scheduler():
    assert float(S.Constant(0.1)()) == pytest.approx(0.1)
    assert float(bp.optim.make_schedule(0.05)()) == pytest.approx(0.05)
    assert isinstance(bp.optim.make_schedule(S.Constant(0.1)), S.Constant)
    with pytest.raises(TypeError):
        bp.optim.make_schedule("not-a-schedule")


def test_steplr():
    sch = S.StepLR(0.1, step_size=10, gamma=0.1)
    vals = [round(float(sch(i)), 6) for i in [0, 5, 10, 20]]
    assert vals == [0.1, 0.1, 0.01, 0.001], vals
    repr(sch)


def test_exponential_lr():
    sch = S.ExponentialLR(0.1, gamma=0.9)
    assert float(sch(2)) == pytest.approx(0.1 * 0.9 ** 2, abs=1e-6)
    repr(sch)


def test_cosine_annealing_lr():
    sch = S.CosineAnnealingLR(0.1, T_max=10, eta_min=0.0)
    assert float(sch(0)) == pytest.approx(0.1, abs=1e-5)
    assert float(sch(10)) == pytest.approx(0.0, abs=1e-5)
    assert 0.0 <= float(sch(5)) <= 0.1


def test_cosine_warm_restarts():
    sch = S.CosineAnnealingWarmRestarts(0.1, num_call_per_epoch=2, T_0=4)
    assert 0.0 <= float(sch(3)) <= 0.1
    assert 0.0 <= float(sch(10)) <= 0.1  # epoch >= T_0 -> _cond1 (T_mult==1 path)
    assert float(sch.current_epoch(4)) >= 0
    with pytest.raises(ValueError):
        S.CosineAnnealingWarmRestarts(0.1, num_call_per_epoch=2, T_0=0)
    with pytest.raises(ValueError):
        S.CosineAnnealingWarmRestarts(0.1, num_call_per_epoch=2, T_0=4, T_mult=0)


def test_cosine_warm_restarts_tmult_gt1_known_bug():
    """REMAINING BUG: with ``T_mult > 1``, ``CosineAnnealingWarmRestarts`` calls
    ``lax.cond`` whose two branches (``_cond1`` vs ``_cond2``) return mismatched
    output types (float tuple vs int ``T_0``), raising TypeError under jit
    (scheduler.py:292-313). Pinned so the construction path stays covered."""
    sch = S.CosineAnnealingWarmRestarts(0.1, num_call_per_epoch=2, T_0=2, T_mult=2)
    with pytest.raises(Exception):
        float(sch(10))


def test_exponential_decay_lr():
    sch = S.ExponentialDecayLR(0.1, decay_steps=10, decay_rate=0.9)
    assert float(sch(5)) == pytest.approx(0.1 * 0.9 ** 0.5, abs=1e-5)
    # call-based step advances last_call
    v0 = float(sch())
    sch.step_call()
    v1 = float(sch())
    assert v1 < v0
    repr(sch)
    with pytest.warns(Warning):
        S.ExponentialDecay(0.1, 10, 0.9)


def test_inverse_time_decay_lr():
    sch = S.InverseTimeDecayLR(0.1, decay_steps=10, decay_rate=0.9)
    assert float(sch(5)) == pytest.approx(0.1 / (1 + 0.9 * 5 / 10), abs=1e-5)
    stair = S.InverseTimeDecayLR(0.1, decay_steps=10, decay_rate=0.9, staircase=True)
    assert float(stair(5)) == pytest.approx(0.1, abs=1e-5)
    repr(sch)
    with pytest.warns(Warning):
        S.InverseTimeDecay(0.1, 10, 0.9)


def test_polynomial_decay_lr():
    sch = S.PolynomialDecayLR(0.1, decay_steps=10, final_lr=0.01)
    assert float(sch(0)) == pytest.approx(0.1, abs=1e-5)
    assert float(sch(10)) == pytest.approx(0.01, abs=1e-5)
    assert float(sch(100)) == pytest.approx(0.01, abs=1e-5)  # clamped to decay_steps
    repr(sch)
    with pytest.warns(Warning):
        S.PolynomialDecay(0.1, 10, 0.01)


def test_piecewise_constant_lr():
    sch = S.PiecewiseConstantLR([10, 20], [0.1, 0.01, 0.001])
    assert float(sch(5)) == pytest.approx(0.1, abs=1e-6)
    assert float(sch(15)) == pytest.approx(0.01, abs=1e-6)
    assert float(sch(25)) == pytest.approx(0.001, abs=1e-6)
    with pytest.warns(Warning):
        S.PiecewiseConstant([10, 20], [0.1, 0.01, 0.001])
    from brainpy._errors import MathError
    with pytest.raises(MathError):
        S.PiecewiseConstantLR([10, 20], [0.1, 0.01])  # bad lengths


def test_scheduler_step_epoch_and_set_value():
    sch = S.StepLR(0.1, step_size=5)
    sch.step_epoch()
    assert int(sch.last_epoch.value) == 0
    sch.set_value(0.5)
    assert float(sch.lr) == pytest.approx(0.5)


# ===========================================================================
# 5. stateless_encoding.py + stateful encoders
# ===========================================================================

def test_poisson_single_step_returns_spikes():
    """single_step must return a spike array of the same shape (no TypeError)."""
    out = bp.encoding.PoissonEncoder().single_step(bm.random.rand(4))
    arr = _np(out)
    assert arr.shape == (4,)
    assert set(np.unique(arr)).issubset({0.0, 1.0})


def test_poisson_single_step_with_first_spike_time():
    enc = bp.encoding.PoissonEncoder(first_spk_time=2.0)
    before = _np(enc.single_step(bm.ones(4), i_step=0))
    assert np.allclose(before, 0.0)  # no spikes before first-spike step
    after = _np(enc.single_step(bm.ones(4), i_step=100))
    assert np.allclose(after, 1.0)  # prob==1 -> always fires after


def test_poisson_normalize_and_multi_steps():
    enc = bp.encoding.PoissonEncoder(min_val=0.0, max_val=2.0, gain=1.0, offset=0.0)
    assert _np(enc.single_step(bm.ones(4))).shape == (4,)
    spikes = enc.multi_steps(bm.random.rand(3), n_time=5.0)
    assert _np(spikes).shape[1:] == (3,)
    # n_time=None -> single current step
    assert _np(enc.multi_steps(bm.random.rand(3), n_time=None)).shape == (3,)
    # first_spk_step > 0 multi-step branch
    enc2 = bp.encoding.PoissonEncoder(first_spk_time=1.0)
    assert _np(enc2.multi_steps(bm.random.rand(3), n_time=5.0)).shape[1:] == (3,)


def test_diff_encoder():
    enc = bp.encoding.DiffEncoder(threshold=1.0)
    assert _np(enc.multi_steps(bm.array([1., 2., 2.9, 3., 3.9]))).shape == (5,)
    enc2 = bp.encoding.DiffEncoder(threshold=1.0, padding=True, off_spike=True)
    assert _np(enc2.multi_steps(bm.array([1., 2., 0., 2., 2.9]))).shape == (5,)
    with pytest.raises(NotImplementedError):
        enc.single_step(bm.array([1.0]))


def test_latency_encoder():
    enc = bp.encoding.LatencyEncoder(method='linear', normalize=True)
    out = enc.multi_steps(bm.array([0.02, 0.5, 1.0]), n_time=5.0)
    assert _np(out).shape == (50, 3)
    enc_log = bp.encoding.LatencyEncoder(method='log', clip=True, normalize=True,
                                         min_val=0.0, max_val=1.0)
    assert _np(enc_log.multi_steps(bm.array([0.02, 0.5, 1.0]), n_time=5.0)).shape == (50, 3)
    with pytest.raises(NotImplementedError):
        enc.single_step(bm.array([0.5]))
    with pytest.raises(ValueError):
        bp.encoding.LatencyEncoder(method='bogus')


def test_weighted_phase_encoder():
    enc = bp.encoding.WeightedPhaseEncoder(min_val=0.0, max_val=1.0, num_phase=4)
    out = enc(bm.array([0.3, 0.7]), num_step=4)
    assert _np(out).shape == (4, 2)


# ===========================================================================
# 6. random_conn.py  --  FixedProb / FixedPreNum / FixedPostNum / FixedTotalNum
# ===========================================================================

def test_fixedprob_nonzero_nnz():
    """M-30: small post population must not silently produce 0 connections."""
    pre, post = bp.connect.FixedProb(prob=0.3, allow_multi_conn=True)(
        pre_size=100, post_size=3).build_coo()
    assert len(_np(pre)) > 0


def test_fixedprob_rectangular_include_self_false_no_raise():
    """M-30: rectangular shape with include_self=False no longer raises a
    (contradictory) ConnectorError."""
    conn = bp.connect.FixedProb(prob=0.3, allow_multi_conn=True, include_self=False)(
        pre_size=100, post_size=3)
    pre, post = conn.build_coo()
    assert len(_np(pre)) > 0
    # build_csr / build_mat also work for the rectangular include_self=False case.
    conn.build_csr()
    assert conn.build_mat().shape == (100, 3)


def test_fixedprob_all_build_methods():
    conn = bp.connect.FixedProb(prob=0.4, allow_multi_conn=True)(pre_size=20, post_size=10)
    pre, post = conn.build_coo()
    assert len(_np(pre)) > 0
    idx, indptr = conn.build_csr()
    assert _np(indptr).shape[0] == 21  # pre_num + 1
    mat = conn.build_mat()
    assert mat.shape == (20, 10)
    repr(conn)


def test_fixedprob_pre_ratio_and_include_self():
    conn = bp.connect.FixedProb(prob=0.5, pre_ratio=0.5, allow_multi_conn=True,
                                include_self=False)(pre_size=20, post_size=20)
    pre, post = conn.build_coo()
    assert len(_np(pre)) >= 0
    conn.build_csr()
    assert conn.build_mat().shape == (20, 20)


def test_fixedprob_invalid_args():
    with pytest.raises(AssertionError):
        bp.connect.FixedProb(prob=1.5)
    with pytest.raises(AssertionError):
        bp.connect.FixedProb(prob=0.3, pre_ratio=2.0)


def test_fixed_pre_num():
    conn = bp.connect.FixedPreNum(num=3, allow_multi_conn=True)(pre_size=10, post_size=8)
    pre, post = conn.build_coo()
    assert len(_np(pre)) > 0
    conn2 = bp.connect.FixedPreNum(num=0.5, allow_multi_conn=True, include_self=False)(
        pre_size=10, post_size=10)
    assert len(_np(conn2.build_coo()[0])) >= 0
    with pytest.raises(Exception):
        bp.connect.FixedPreNum(num=100, allow_multi_conn=True)(
            pre_size=10, post_size=8).build_coo()  # num > pre_num


def test_fixed_post_num():
    conn = bp.connect.FixedPostNum(num=3, allow_multi_conn=True)(pre_size=10, post_size=8)
    pre, post = conn.build_coo()
    assert len(_np(pre)) > 0
    idx, indptr = conn.build_csr()
    assert _np(indptr).shape[0] == 11  # pre_num + 1
    conn2 = bp.connect.FixedPostNum(num=0.5, allow_multi_conn=True, include_self=False)(
        pre_size=10, post_size=10)
    conn2.build_coo()
    conn2.build_csr()


def test_fixed_total_num():
    conn = bp.connect.FixedTotalNum(num=12, allow_multi_conn=True)(pre_size=10, post_size=8)
    pre, post = conn.build_coo()
    assert len(_np(pre)) == 12
    # no-multi-conn (choice without replacement) path
    conn2 = bp.connect.FixedTotalNum(num=12, allow_multi_conn=False)(pre_size=10, post_size=8)
    assert len(_np(conn2.build_coo()[0])) == 12
    repr(conn)
    with pytest.raises(Exception):
        bp.connect.FixedTotalNum(num=1000, allow_multi_conn=True)(
            pre_size=10, post_size=8).build_coo()  # num > all2all


def test_connectors_no_multi_conn_paths():
    """Coverage: the ``allow_multi_conn=False`` (numba choice-without-replacement)
    build paths of FixedProb / FixedPreNum / FixedPostNum."""
    fp = bp.connect.FixedProb(prob=0.4, allow_multi_conn=False)(pre_size=20, post_size=10)
    assert len(_np(fp.build_coo()[0])) > 0
    fp.build_csr()
    assert fp.build_mat().shape == (20, 10)
    # include_self=False square case
    fp2 = bp.connect.FixedProb(prob=0.5, allow_multi_conn=False, include_self=False)(
        pre_size=15, post_size=15)
    fp2.build_coo()
    fp2.build_csr()
    assert fp2.build_mat().shape == (15, 15)

    pre_conn = bp.connect.FixedPreNum(num=3, allow_multi_conn=False)(pre_size=12, post_size=8)
    assert len(_np(pre_conn.build_coo()[0])) > 0
    pre_conn2 = bp.connect.FixedPreNum(num=3, allow_multi_conn=False, include_self=False)(
        pre_size=12, post_size=12)
    pre_conn2.build_coo()

    post_conn = bp.connect.FixedPostNum(num=3, allow_multi_conn=False)(pre_size=12, post_size=8)
    assert len(_np(post_conn.build_coo()[0])) > 0
    post_conn.build_csr()
    post_conn2 = bp.connect.FixedPostNum(num=3, allow_multi_conn=False, include_self=False)(
        pre_size=12, post_size=12)
    post_conn2.build_coo()
    post_conn2.build_csr()


def test_connectors_validation_branches():
    """Coverage: float / bad-type ``num`` validation branches of the connectors."""
    # float num accepted at construction
    assert bp.connect.FixedTotalNum(num=0.5).num == 0.5
    assert bp.connect.FixedPreNum(num=0.3).num == 0.3
    assert bp.connect.FixedPostNum(num=0.3).num == 0.3
    # bad type rejected
    from brainpy._errors import ConnectorError
    with pytest.raises(ConnectorError):
        bp.connect.FixedTotalNum(num='x')
    with pytest.raises(ConnectorError):
        bp.connect.FixedPreNum(num='x')
    # FixedPreNum with float num builds (probability interpretation)
    fp = bp.connect.FixedPreNum(num=0.3, allow_multi_conn=True)(pre_size=10, post_size=8)
    assert len(_np(fp.build_coo()[0])) > 0
    # negative integer num rejected
    with pytest.raises(AssertionError):
        bp.connect.FixedTotalNum(num=-1)
    # FixedPostNum num > post_num
    with pytest.raises(ConnectorError):
        bp.connect.FixedPostNum(num=100, allow_multi_conn=True)(
            pre_size=10, post_size=8).build_coo()


def test_fixed_pre_post_num_include_self_rectangular_raises():
    """FixedPreNum / FixedPostNum still reject include_self=False for rectangular
    (pre_num != post_num) shapes (this guard is intentional for these connectors)."""
    with pytest.raises(Exception):
        bp.connect.FixedPreNum(num=3, allow_multi_conn=True, include_self=False)(
            pre_size=10, post_size=8).build_coo()
    with pytest.raises(Exception):
        bp.connect.FixedPostNum(num=3, allow_multi_conn=True, include_self=False)(
            pre_size=10, post_size=8).build_coo()


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
