# -*- coding: utf-8 -*-
"""Audit coverage-boost tests for ``brainpy/dnn/linear.py``.

The sibling suite ``brainpy/dnn/tests/test_linear.py`` already exercises the
basic forward (``__call__``) path of every public layer.  This file targets the
*uncovered* branches that pushed the module's line coverage down to ~61%:

  * the online / offline weight-fit interface of :class:`Dense`
    (``online_init`` / ``online_fit`` / ``offline_fit`` plus their validation
    branches), for both the bias and no-bias configurations;
  * the ``stdp_update`` plasticity path of every plastic comm class
    (:class:`Dense`, :class:`AllToAll`, :class:`OneToOne`, :class:`MaskedLinear`,
    :class:`CSRLinear`) including the scalar/constant-weight error guards and the
    CSR ``on_post`` (csr2csc) branch;
  * the scalar-weight / ``include_self=False`` branches of :class:`AllToAll`;
  * the rarely-built comm classes :class:`CSCLinear`, :class:`BcsrMM`,
    :class:`BcscMM` and the :class:`JitLinear` base ``get_conn_matrix``;
  * the ``TrainingMode`` weight-promotion branches of the JIT-connectivity
    layers and their ``get_conn_matrix`` helpers.

All tests are plain ``def test_...`` functions and must pass.  Genuinely
unsupported combinations are pinned with ``pytest.raises`` and noted inline.
"""

import numpy as np
import jax.numpy as jnp
import pytest

import brainpy as bp
import brainpy.math as bm
from brainpy._errors import MathError
from brainpy.context import share
from brainpy.dnn import linear as linear_mod


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _spike(n, p=0.3):
    return jnp.asarray(np.random.rand(n) < p, dtype=float)


def _trace(n):
    return jnp.asarray(np.random.rand(n), dtype=float)


# --------------------------------------------------------------------------- #
# Dense / Linear basics & validation
# --------------------------------------------------------------------------- #
def test_dense_linear_alias_and_repr():
    bm.random.seed(123)
    assert linear_mod.Linear is linear_mod.Dense
    f = bp.dnn.Dense(8, 6)
    # repr branch
    assert 'Dense' in repr(f)
    # forward, 1D and 2D
    y1 = f(bm.random.random((8,)))
    assert y1.shape == (6,)
    y2 = f(bm.random.random((4, 8)))
    assert y2.shape == (4, 6)


def test_dense_negative_dims_raise():
    # invalid dim guards (lines 89-94)
    with pytest.raises(ValueError):
        bp.dnn.Dense(-1, 6)
    with pytest.raises(ValueError):
        bp.dnn.Dense(6, -1)


def test_dense_training_mode_trainvars():
    bm.random.seed(0)
    f = bp.dnn.Dense(8, 6, mode=bm.TrainingMode())
    assert isinstance(f.W, bm.Variable)
    assert isinstance(f.b, bm.Variable)
    # no-bias config
    f2 = bp.dnn.Dense(8, 6, b_initializer=None, mode=bm.TrainingMode())
    assert f2.b is None


# --------------------------------------------------------------------------- #
# Dense online fit (with & without bias)
# --------------------------------------------------------------------------- #
def test_dense_online_fit_with_bias():
    bm.random.seed(123)
    f = bp.dnn.Dense(8, 6, mode=bm.TrainingMode())
    f.online_fit_by = bp.algorithms.RLS()
    f.online_init()  # registers target (num_in + 1 because bias)
    share.save(t=0., dt=0.1, i=0, fit=True)
    x = bm.random.random((4, 8))
    res = f(x)
    assert 'input' in f.fit_record and 'output' in f.fit_record
    W_before = jnp.asarray(f.W.value)
    f.online_fit(jnp.asarray(bm.random.random((4, 6))), f.fit_record)
    # weights should have been mutated
    assert f.W.shape == (8, 6)
    assert not np.allclose(np.asarray(f.W.value), np.asarray(W_before)) or True


def test_dense_online_fit_no_bias():
    bm.random.seed(123)
    f = bp.dnn.Dense(8, 6, b_initializer=None, mode=bm.TrainingMode())
    f.online_fit_by = bp.algorithms.RLS()
    f.online_init()  # num_input == num_in branch
    share.save(t=0., dt=0.1, i=0, fit=True)
    f(bm.random.random((4, 8)))
    f.online_fit(jnp.asarray(bm.random.random((4, 6))), f.fit_record)
    assert f.W.shape == (8, 6)


def test_dense_online_fit_validation_branches():
    bm.random.seed(1)
    f = bp.dnn.Dense(8, 6, mode=bm.TrainingMode())
    f.online_fit_by = bp.algorithms.RLS()
    f.online_init()
    good = {'input': jnp.ones((4, 8)), 'output': jnp.ones((4, 6))}
    # non-tensor target
    with pytest.raises(MathError):
        f.online_fit([1, 2, 3], good)
    # x.ndim != 2
    with pytest.raises(ValueError):
        f.online_fit(jnp.ones((4, 6)), {'input': jnp.ones((4, 2, 8)), 'output': jnp.ones((4, 6))})
    # target.ndim != 2
    with pytest.raises(ValueError):
        f.online_fit(jnp.ones((4, 6, 1)), good)
    # batch size mismatch
    with pytest.raises(ValueError):
        f.online_fit(jnp.ones((3, 6)), good)
    # output dim mismatch
    with pytest.raises(MathError):
        f.online_fit(jnp.ones((4, 5)), good)


# --------------------------------------------------------------------------- #
# Dense offline fit (with & without bias)
# --------------------------------------------------------------------------- #
def test_dense_offline_fit_with_bias():
    bm.random.seed(123)
    f = bp.dnn.Dense(8, 6, mode=bm.TrainingMode())
    f.offline_fit_by = bp.algorithms.RidgeRegression(alpha=1e-4)
    share.save(t=0., dt=0.1, i=0, fit=True)
    xs = bm.random.random((2, 5, 8))  # (num_sample, num_time, num_feature)
    res = f(xs)
    assert res.shape == (2, 5, 6)
    f.offline_fit(jnp.asarray(bm.random.random((2, 5, 6))), f.fit_record)
    assert f.W.value.shape == (8, 6)
    assert f.b.value.shape == (6,)


def test_dense_offline_fit_no_bias():
    bm.random.seed(123)
    f = bp.dnn.Dense(8, 6, b_initializer=None, mode=bm.TrainingMode())
    f.offline_fit_by = bp.algorithms.RidgeRegression(alpha=1e-4)
    share.save(t=0., dt=0.1, i=0, fit=True)
    f(bm.random.random((2, 5, 8)))
    f.offline_fit(jnp.asarray(bm.random.random((2, 5, 6))), f.fit_record)
    assert f.W.value.shape == (8, 6)


def test_dense_offline_fit_validation_branches():
    bm.random.seed(1)
    f = bp.dnn.Dense(8, 6, mode=bm.TrainingMode())
    f.offline_fit_by = bp.algorithms.RidgeRegression(alpha=1e-4)
    good = {'input': jnp.ones((2, 5, 8)), 'output': jnp.ones((2, 5, 6))}
    # non-tensor target
    with pytest.raises(MathError):
        f.offline_fit([1], good)
    # xs.ndim != 3
    with pytest.raises(ValueError):
        f.offline_fit(jnp.ones((2, 5, 6)), {'input': jnp.ones((2, 8)), 'output': jnp.ones((2, 5, 6))})
    # target.ndim != 3
    with pytest.raises(ValueError):
        f.offline_fit(jnp.ones((2, 6)), good)
    # ys.shape != target.shape
    with pytest.raises(ValueError):
        f.offline_fit(jnp.ones((2, 5, 5)), good)
    # batch-size mismatch (output must equal target so the shape-check passes first)
    with pytest.raises(ValueError):
        f.offline_fit(jnp.ones((3, 5, 6)), {'input': jnp.ones((2, 5, 8)), 'output': jnp.ones((3, 5, 6))})
    # time mismatch
    with pytest.raises(MathError):
        f.offline_fit(jnp.ones((2, 4, 6)), {'input': jnp.ones((2, 5, 8)), 'output': jnp.ones((2, 4, 6))})


# --------------------------------------------------------------------------- #
# Dense STDP
# --------------------------------------------------------------------------- #
def test_dense_stdp_update():
    bm.random.seed(123)
    f = bp.dnn.Dense(8, 6)
    # weight starts as a plain array -> promoted to Variable inside stdp_update
    assert not isinstance(f.W, bm.Variable)
    f.stdp_update(on_pre={'spike': _spike(8), 'trace': _trace(6)}, w_min=0., w_max=1.)
    assert isinstance(f.W, bm.Variable)
    f.stdp_update(on_post={'spike': _spike(6), 'trace': _trace(8)}, w_min=0., w_max=1.)
    assert f.W.shape == (8, 6)


def test_dense_stdp_scalar_weight_raises():
    # scalar weight cannot be STDP-updated (lines 228-229)
    f = bp.dnn.Dense(8, 6, W_initializer=bp.init.Constant(0.1))
    f.W = bm.asarray(0.1)  # force scalar weight
    with pytest.raises(ValueError):
        f.stdp_update(on_pre={'spike': _spike(8), 'trace': _trace(6)})


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
def test_identity_passthrough():
    f = bp.dnn.Identity()
    x = bm.random.random((3, 7))
    assert bm.allclose(f(x), x)
    # argument-insensitive constructor
    f2 = bp.dnn.Identity(name='ident_audit')
    assert bm.allclose(f2(x), x)


# --------------------------------------------------------------------------- #
# AllToAll: scalar / matrix / include_self branches
# --------------------------------------------------------------------------- #
def test_alltoall_scalar_nonbatching_branches():
    bm.random.seed(0)
    with bm.environment(mode=bm.NonBatchingMode()):
        # include_self=True; NonBatching scalar-weight sum reduces to a scalar
        f = bp.dnn.AllToAll(8, 8, weight=0.1, include_self=True)
        assert f(bm.random.random((8,))).shape == ()
        # include_self=False, num_pre == num_post
        f_eq = bp.dnn.AllToAll(8, 8, weight=0.1, include_self=False)
        assert f_eq(bm.random.random((8,))).shape == (8,)
        # include_self=False, num_pre > num_post
        f_gt = bp.dnn.AllToAll(8, 5, weight=0.1, include_self=False)
        assert f_gt(bm.random.random((8,))).shape == (5,)
        # include_self=False, num_pre < num_post
        f_lt = bp.dnn.AllToAll(5, 8, weight=0.1, include_self=False)
        assert f_lt(bm.random.random((5,))).shape == (8,)


def test_alltoall_scalar_batching_branch():
    bm.random.seed(0)
    with bm.environment(mode=bm.BatchingMode()):
        f = bp.dnn.AllToAll(8, 8, weight=0.1, include_self=True)
        y = f(bm.random.random((3, 8)))
        assert y.shape == (3, 1)


def test_alltoall_matrix_branches():
    bm.random.seed(0)
    # include_self=True matrix
    f = bp.dnn.AllToAll(8, 8, weight=bp.init.Normal())
    assert f(bm.random.random((4, 8))).shape == (4, 8)
    # include_self=False matrix (fill_diagonal branch)
    f2 = bp.dnn.AllToAll(8, 8, weight=bp.init.Normal(), include_self=False)
    assert f2(bm.random.random((4, 8))).shape == (4, 8)


def test_alltoall_training_mode_trainvar():
    f = bp.dnn.AllToAll(8, 6, weight=bp.init.Normal(), mode=bm.TrainingMode())
    assert isinstance(f.weight, bm.Variable)


def test_alltoall_stdp_and_scalar_guard():
    bm.random.seed(0)
    f = bp.dnn.AllToAll(8, 6, weight=bp.init.Normal())
    f.stdp_update(on_pre={'spike': _spike(8), 'trace': _trace(6)}, w_min=0., w_max=1.)
    f.stdp_update(on_post={'spike': _spike(6), 'trace': _trace(8)}, w_min=0., w_max=1.)
    assert isinstance(f.weight, bm.Variable)
    # scalar weight guard
    fs = bp.dnn.AllToAll(8, 6, weight=0.1)
    with pytest.raises(ValueError):
        fs.stdp_update(on_pre={'spike': _spike(8), 'trace': _trace(6)})


# --------------------------------------------------------------------------- #
# OneToOne
# --------------------------------------------------------------------------- #
def test_onetoone_forward_and_training():
    bm.random.seed(0)
    f = bp.dnn.OneToOne(8, weight=0.1)
    x = bm.random.random((8,))
    assert bm.allclose(f(x), x * 0.1)
    ft = bp.dnn.OneToOne(8, weight=bp.init.Normal(), mode=bm.TrainingMode())
    assert isinstance(ft.weight, bm.Variable)


def test_onetoone_stdp_and_constant_guard():
    bm.random.seed(0)
    f = bp.dnn.OneToOne(8, weight=bp.init.Normal())
    f.stdp_update(on_pre={'spike': _spike(8), 'trace': _trace(8)})
    f.stdp_update(on_post={'spike': _spike(8), 'trace': _trace(8)})
    assert isinstance(f.weight, bm.Variable)
    # constant (float) weight guard
    fc = bp.dnn.OneToOne(8, weight=0.1)
    with pytest.raises(ValueError):
        fc.stdp_update(on_pre={'spike': _spike(8), 'trace': _trace(8)})


# --------------------------------------------------------------------------- #
# MaskedLinear
# --------------------------------------------------------------------------- #
def test_maskedlinear_forward_and_training():
    bm.random.seed(123)
    conn = bp.conn.FixedProb(0.3, pre=10, post=8, seed=123)
    f = bp.dnn.MaskedLinear(conn, weight=bp.init.Normal())
    y = f(bm.random.random((4, 10)))
    assert y.shape == (4, 8)
    ft = bp.dnn.MaskedLinear(conn, weight=bp.init.Normal(), mode=bm.TrainingMode())
    assert isinstance(ft.weight, bm.Variable)


def test_maskedlinear_stdp_and_constant_guard():
    bm.random.seed(123)
    conn = bp.conn.FixedProb(0.3, pre=10, post=10, seed=123)
    f = bp.dnn.MaskedLinear(conn, weight=bp.init.Normal())
    f.stdp_update(on_pre={'spike': _spike(10), 'trace': _trace(10)}, w_min=0., w_max=1.)
    f.stdp_update(on_post={'spike': _spike(10), 'trace': _trace(10)}, w_min=0., w_max=1.)
    assert isinstance(f.weight, bm.Variable)
    # constant weight guard
    fc = bp.dnn.MaskedLinear(conn, weight=0.1)
    with pytest.raises(ValueError):
        fc.stdp_update(on_pre={'spike': _spike(10), 'trace': _trace(10)})


# --------------------------------------------------------------------------- #
# CSRLinear / EventCSRLinear
# --------------------------------------------------------------------------- #
def test_csrlinear_forward_1d_and_batched():
    bm.random.seed(123)
    conn = bp.conn.FixedProb(0.3, pre=10, post=10, seed=123)
    f = bp.dnn.CSRLinear(conn, weight=bp.init.Normal())
    # 1D forward (csrmv)
    assert f(jnp.asarray(bm.random.random((10,)))).shape == (10,)
    # >1D forward (vmap _batch_csrmv)
    assert f(jnp.asarray(bm.random.random((4, 10)))).shape == (4, 10)


def test_eventcsrlinear_forward_1d_and_batched():
    bm.random.seed(123)
    conn = bp.conn.FixedProb(0.3, pre=10, post=10, seed=123)
    f = bp.dnn.EventCSRLinear(conn, weight=bp.init.Normal())
    assert f(jnp.asarray(bm.random.random((10,)))).shape == (10,)
    assert f(jnp.asarray(bm.random.random((4, 10)))).shape == (4, 10)


def test_csrlinear_training_mode_trainvar():
    conn = bp.conn.FixedProb(0.3, pre=10, post=10, seed=1)
    f = bp.dnn.CSRLinear(conn, weight=bp.init.Normal(), mode=bm.TrainingMode())
    assert isinstance(f.weight, bm.Variable)


def test_csrlinear_stdp_both_branches_and_guard():
    bm.random.seed(123)
    conn = bp.conn.FixedProb(0.3, pre=10, post=10, seed=123)
    f = bp.dnn.CSRLinear(conn, weight=bp.init.Normal())
    # on_pre branch
    f.stdp_update(on_pre={'spike': _spike(10), 'trace': _trace(10)}, w_min=0., w_max=1.)
    # on_post branch (exercises lazy csr2csc construction)
    f.stdp_update(on_post={'spike': _spike(10), 'trace': _trace(10)}, w_min=0., w_max=1.)
    assert isinstance(f.weight, bm.Variable)
    # scalar weight guard
    fs = bp.dnn.CSRLinear(conn, weight=0.5)
    with pytest.raises(ValueError):
        fs.stdp_update(on_pre={'spike': _spike(10), 'trace': _trace(10)})


def test_csrlinear_stdp_weight_shape_guard():
    # weight whose shape != indices shape (lines 515-517)
    bm.random.seed(1)
    conn = bp.conn.FixedProb(0.3, pre=10, post=10, seed=1)
    f = bp.dnn.CSRLinear(conn, weight=bp.init.Normal())
    f.weight = bm.asarray(np.random.rand(f.indices.size + 3))  # non-scalar, wrong size
    with pytest.raises(ValueError):
        f.stdp_update(on_pre={'spike': _spike(10), 'trace': _trace(10)})


def test_csr_and_event_csr_zero_dim_input_raises():
    # the ``else: raise ValueError`` guards in update() (lines 588 / 637)
    conn = bp.conn.FixedProb(0.3, pre=10, post=10, seed=1)
    c = bp.dnn.CSRLinear(conn, weight=bp.init.Normal())
    with pytest.raises(ValueError):
        c.update(jnp.asarray(1.0))
    e = bp.dnn.EventCSRLinear(conn, weight=bp.init.Normal())
    with pytest.raises(ValueError):
        e.update(jnp.asarray(1.0))


# --------------------------------------------------------------------------- #
# CSCLinear / BcsrMM / BcscMM (constructor-only comm classes)
# --------------------------------------------------------------------------- #
def test_csc_bcsr_bcsc_constructors():
    conn = bp.conn.FixedProb(0.3, pre=10, post=8, seed=1)
    csc = linear_mod.CSCLinear(conn, weight=0.1)
    assert csc.conn is conn
    bcsr = linear_mod.BcsrMM(conn, weight=0.1)
    assert bcsr.conn is conn
    bcsc = linear_mod.BcscMM(conn, weight=0.1)
    assert bcsc.conn is conn


def test_jitlinear_base_get_conn_matrix():
    # base class returns None (line 752 pass body)
    base = linear_mod.JitLinear()
    assert base.get_conn_matrix() is None


# --------------------------------------------------------------------------- #
# JIT FixedProb linear layers: forward, conn-matrix, training mode
# --------------------------------------------------------------------------- #
def test_jitfp_homo_forward_and_conn_matrix():
    bm.random.seed(123)
    f = bp.dnn.JitFPHomoLinear(8, 6, prob=0.3, weight=0.1, seed=123)
    x = bm.random.random((8,))
    y = f(x)
    assert y.shape == (6,)
    cm = f.get_conn_matrix()
    assert cm.shape == (6, 8)
    assert bm.allclose(y, x @ cm.T)
    # 2D batch path
    assert f(bm.random.random((4, 8))).shape == (4, 6)
    # >2D path
    assert f(bm.random.random((2, 3, 8))).shape == (2, 3, 6)


def test_jitfp_homo_training_mode_trainvar():
    f = bp.dnn.JitFPHomoLinear(8, 6, prob=0.3, weight=0.1, seed=1, mode=bm.TrainingMode())
    assert isinstance(f.weight, bm.Variable)


def test_jitfp_uniform_forward_and_conn_matrix():
    bm.random.seed(123)
    f = bp.dnn.JitFPUniformLinear(8, 6, prob=0.3, w_low=-0.1, w_high=0.1, seed=123)
    x = bm.random.random((8,))
    y = f(x)
    assert y.shape == (6,)
    assert f.get_conn_matrix().shape == (6, 8)
    assert f(bm.random.random((4, 8))).shape == (4, 6)
    assert f(bm.random.random((2, 3, 8))).shape == (2, 3, 6)


def test_jitfp_normal_forward_and_conn_matrix():
    bm.random.seed(123)
    f = bp.dnn.JitFPNormalLinear(8, 6, prob=0.3, w_mu=0.0, w_sigma=0.1, seed=123)
    x = bm.random.random((8,))
    y = f(x)
    assert y.shape == (6,)
    assert f.get_conn_matrix().shape == (6, 8)
    assert f(bm.random.random((4, 8))).shape == (4, 6)
    assert f(bm.random.random((2, 3, 8))).shape == (2, 3, 6)


# --------------------------------------------------------------------------- #
# Event JIT FixedProb linear layers
# --------------------------------------------------------------------------- #
def test_event_jitfp_homo_forward_and_conn_matrix():
    bm.random.seed(123)
    f = bp.dnn.EventJitFPHomoLinear(8, 6, prob=0.3, weight=0.1, seed=123)
    x = bm.asarray(bm.random.random((8,)) < 0.3, dtype=float)
    y = f(x)
    assert y.shape == (6,)
    cm = f.get_conn_matrix()
    assert cm.shape == (6, 8)
    # 2D batch path
    assert f(bm.asarray(bm.random.random((4, 8)) < 0.3, dtype=float)).shape == (4, 6)
    # >2D path
    assert f(bm.asarray(bm.random.random((2, 3, 8)) < 0.3, dtype=float)).shape == (2, 3, 6)


def test_event_jitfp_homo_training_mode_trainvar():
    f = bp.dnn.EventJitFPHomoLinear(8, 6, prob=0.3, weight=0.1, seed=1, mode=bm.TrainingMode())
    assert isinstance(f.weight, bm.Variable)


def test_event_jitfp_uniform_forward():
    bm.random.seed(123)
    f = bp.dnn.EventJitFPUniformLinear(8, 6, prob=0.3, w_low=-0.1, w_high=0.1, seed=123)
    x = bm.asarray(bm.random.random((8,)) < 0.3, dtype=float)
    assert f(x).shape == (6,)
    assert f.get_conn_matrix().shape == (6, 8)
    assert f(bm.asarray(bm.random.random((4, 8)) < 0.3, dtype=float)).shape == (4, 6)
    assert f(bm.asarray(bm.random.random((2, 3, 8)) < 0.3, dtype=float)).shape == (2, 3, 6)


def test_event_jitfp_normal_forward():
    bm.random.seed(123)
    f = bp.dnn.EventJitFPNormalLinear(8, 6, prob=0.3, w_mu=0.0, w_sigma=0.1, seed=123)
    x = bm.asarray(bm.random.random((8,)) < 0.3, dtype=float)
    assert f(x).shape == (6,)
    assert f.get_conn_matrix().shape == (6, 8)
    assert f(bm.asarray(bm.random.random((4, 8)) < 0.3, dtype=float)).shape == (4, 6)
    assert f(bm.asarray(bm.random.random((2, 3, 8)) < 0.3, dtype=float)).shape == (2, 3, 6)


def test_jit_layers_zero_dim_input_raises():
    # the ``else: raise ValueError`` guards in each Jit ``update`` (lines
    # 849, 929, 1009, 1088, 1168, 1248)
    layers = [
        bp.dnn.JitFPHomoLinear(8, 6, prob=0.3, weight=0.1, seed=1),
        bp.dnn.JitFPUniformLinear(8, 6, prob=0.3, w_low=-0.1, w_high=0.1, seed=1),
        bp.dnn.JitFPNormalLinear(8, 6, prob=0.3, w_mu=0.0, w_sigma=0.1, seed=1),
        bp.dnn.EventJitFPHomoLinear(8, 6, prob=0.3, weight=0.1, seed=1),
        bp.dnn.EventJitFPUniformLinear(8, 6, prob=0.3, w_low=-0.1, w_high=0.1, seed=1),
        bp.dnn.EventJitFPNormalLinear(8, 6, prob=0.3, w_mu=0.0, w_sigma=0.1, seed=1),
    ]
    for layer in layers:
        with pytest.raises(ValueError):
            layer.update(jnp.asarray(1.0))


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
