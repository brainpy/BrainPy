# -*- coding: utf-8 -*-
"""Regression + coverage tests for the ``brainpy.math`` compatibility layer.

This module accompanies the audit recorded in ``docs/issues-found-20260618.md``.
It pins the behaviours fixed by the audit and broadly exercises the numpy /
pytorch / tensorflow compatibility shims, the activation functions, the misc.
``others`` helpers, the ``_utils`` wrapper factory and the ``einops`` port.

Audit findings exercised here (see the doc for full context):

* C-11 (``compat_tensorflow.py``) -- ``reduce_logsumexp`` must be numerically
  stable (delegated to ``jax.scipy.special.logsumexp``).
* H-16 (``activations.py``)        -- ``softmin`` must subtract the max so it
  stays finite for large inputs.
* H-14 (``_utils.py`` + pytorch)   -- ``out=`` wrapped funcs must *return* the
  ``out`` Array instead of ``None``.
* H-15 (``others.py``)             -- ``remove_diag`` must trace cleanly under
  ``jit``/``vmap`` (static off-diagonal gather, handles non-square).
* H-13 (``compat_numpy.py``)       -- ``asfarray`` must coerce integer input to
  a floating dtype.
* M-11/M-12 (``compat_numpy.py``)  -- ``empty`` uses ``jnp.empty`` and
  ``fill_diagonal(inplace=False)`` returns a brainpy ``Array``.
* (``compat_pytorch.py``)          -- ``arcsinh``/``arctanh`` exist & correct,
  no duplicate ``arcsin`` clobbering.
* (``einops.py``)                  -- module still imports after the dead
  ``_optimize_transformation`` helper was removed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import brainpy.math as bm
from brainpy.math import (
    compat_numpy as cn,
    compat_pytorch as cpt,
    compat_tensorflow as ctf,
    activations as act,
    others as bo,
    _utils as butils,
    einops as bein,
)
from brainpy.math.ndarray import Array


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _j(x):
    """Return the underlying jax array for assertions."""
    return bm.as_jax(x)


def _finite(x):
    return bool(jnp.all(jnp.isfinite(_j(x))))


# ===========================================================================
# 1. Regression tests (audit-specific behaviours)
# ===========================================================================

# --- C-11: reduce_logsumexp numerical stability -----------------------------

def test_reduce_logsumexp_stable_large_inputs():
    """C-11: log(sum(exp(.))) must not overflow for large inputs."""
    r = ctf.reduce_logsumexp(bm.asarray([1000., 1000., 1000.]))
    assert _finite(r)
    # logsumexp([1000]*3) == 1000 + log(3) == 1001.0986...
    assert float(_j(r)) == pytest.approx(1000.0 + np.log(3.0), abs=1e-2)


def test_reduce_logsumexp_matches_reference_small():
    x = bm.asarray([0.5, -1.0, 2.0, 3.5])
    r = ctf.reduce_logsumexp(x)
    ref = jax.scipy.special.logsumexp(_j(x))
    assert float(_j(r)) == pytest.approx(float(ref), rel=1e-6)


def test_reduce_logsumexp_axis_keepdims():
    x = bm.asarray([[1., 2., 3.], [4., 5., 6.]])
    r = ctf.reduce_logsumexp(x, axis=1, keepdims=True)
    assert _j(r).shape == (2, 1)
    assert _finite(r)


# --- H-16: softmin must not produce NaN for large inputs --------------------

def test_softmin_finite_for_large_inputs():
    """H-16: softmin lacked max-subtraction -> NaN for large inputs."""
    r = act.softmin(bm.asarray([1000., 1001., 1002.]))
    assert _finite(r)
    # softmin == softmax(-x); the smallest input gets the largest weight.
    expected = np.array([0.66524096, 0.24472847, 0.09003057])
    np.testing.assert_allclose(np.asarray(_j(r)), expected, atol=1e-4)
    assert float(jnp.sum(_j(r))) == pytest.approx(1.0, abs=1e-6)


def test_softmin_matches_softmax_of_negative():
    x = bm.asarray([0.3, -1.2, 2.5, 0.0])
    np.testing.assert_allclose(
        np.asarray(_j(act.softmin(x))),
        np.asarray(_j(act.softmax(-x))),
        atol=1e-6,
    )


# --- H-14: out= wrapped funcs must RETURN the out Array ---------------------

def test_numpy_sum_out_returns_array():
    """H-14: numpy-style ``out=`` should return the (filled) Array, not None."""
    out = bm.asarray(0.)
    r = bm.sum(bm.asarray([1., 2., 3.]), out=out)
    assert r is not None
    assert isinstance(r, Array)
    assert r is out
    assert float(_j(r)) == pytest.approx(6.0)
    assert float(_j(out)) == pytest.approx(6.0)


def test_numpy_out_must_be_brainpy_array():
    with pytest.raises(TypeError):
        bm.sum(bm.asarray([1., 2., 3.]), out=jnp.array(0.))


def test_pytorch_add_out_returns_out():
    """H-14 (pytorch compat): ``add(..., out=...)`` returns ``out``."""
    out = bm.zeros((3,))
    r = cpt.add(bm.asarray([1., 2., 3.]), bm.asarray([1., 1., 1.]), out=out)
    assert r is out
    np.testing.assert_allclose(np.asarray(_j(r)), [2., 3., 4.])


def test_pytorch_out_must_be_brainpy_array():
    with pytest.raises(TypeError):
        cpt.abs(bm.asarray([-1.]), out=jnp.array(0.))


# --- H-15: remove_diag must trace cleanly under jit/vmap --------------------

def test_remove_diag_jit_matches_eager_square():
    """H-15: remove_diag must work under jit (static off-diag gather)."""
    x = jnp.arange(9.).reshape(3, 3)
    eager = _j(bo.remove_diag(x))
    jitted = jax.jit(bo.remove_diag)(x)
    assert eager.shape == (3, 2)
    np.testing.assert_allclose(np.asarray(eager), np.asarray(jitted))
    # row 0 has its diagonal (0) removed -> [1, 2]
    np.testing.assert_allclose(np.asarray(eager[0]), [1., 2.])


def test_remove_diag_vmap():
    x = jnp.arange(2 * 9.).reshape(2, 3, 3)
    out = jax.vmap(bo.remove_diag)(x)
    assert out.shape == (2, 3, 2)
    np.testing.assert_allclose(np.asarray(out[0]), np.asarray(_j(bo.remove_diag(x[0]))))


def test_remove_diag_non_square():
    x = jnp.arange(12.).reshape(3, 4)
    out = _j(bo.remove_diag(x))
    assert out.shape == (3, 3)
    # row 0 drops col 0; rows keep all but the diagonal element
    np.testing.assert_allclose(np.asarray(out[0]), [1., 2., 3.])
    np.testing.assert_allclose(np.asarray(out[1]), [4., 6., 7.])


def test_remove_diag_rejects_non_2d():
    with pytest.raises(ValueError):
        bo.remove_diag(jnp.arange(3.))


# --- H-13: asfarray coerces integers to a floating dtype --------------------

def test_asfarray_integer_input_becomes_float():
    """H-13: asfarray(int) used to no-op; must yield a floating dtype."""
    r = cn.asfarray([1, 2, 3])
    assert jnp.issubdtype(r.dtype, jnp.floating)
    np.testing.assert_allclose(np.asarray(_j(r)), [1., 2., 3.])


def test_asfarray_preserves_floating_dtype():
    r = cn.asfarray(jnp.array([1., 2.], dtype=jnp.float32))
    assert r.dtype == jnp.float32


# --- M-11 / M-12: empty + fill_diagonal -------------------------------------

def test_empty_shape_and_type():
    """M-11: empty must produce the right shape/dtype as a brainpy Array."""
    e = bm.empty((2, 3))
    assert isinstance(e, Array)
    assert e.shape == (2, 3)
    assert jnp.issubdtype(e.dtype, jnp.floating)


def test_empty_like():
    a = bm.asarray(jnp.ones((4,), dtype=jnp.int32))
    e = cn.empty_like(a)
    assert e.shape == (4,)
    assert e.dtype == jnp.int32


def test_fill_diagonal_not_inplace_returns_array():
    """M-12: fill_diagonal(inplace=False) must return a brainpy Array."""
    x = bm.asarray(jnp.ones((3, 3)))
    r = cn.fill_diagonal(x, 5., inplace=False)
    assert isinstance(r, Array)
    np.testing.assert_allclose(np.diag(np.asarray(_j(r))), [5., 5., 5.])
    # original unchanged
    np.testing.assert_allclose(np.diag(np.asarray(_j(x))), [1., 1., 1.])


def test_fill_diagonal_inplace_updates_array():
    x = bm.asarray(jnp.ones((3, 3)))
    out = cn.fill_diagonal(x, 7., inplace=True)
    assert out is None  # in-place returns nothing
    np.testing.assert_allclose(np.diag(np.asarray(_j(x))), [7., 7., 7.])


def test_fill_diagonal_errors():
    with pytest.raises(ValueError):
        cn.fill_diagonal(bm.asarray(jnp.arange(3)), 1.)  # ndim < 2
    with pytest.raises(ValueError):
        cn.fill_diagonal(jnp.ones((3, 3)), 1.)  # inplace on non-Array


# --- pytorch arcsinh / arctanh exist & correct, no dup arcsin ---------------

def test_pytorch_arcsinh_arctanh():
    np.testing.assert_allclose(
        np.asarray(_j(cpt.arcsinh(bm.asarray([0., 1.])))),
        np.arcsinh([0., 1.]), atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(_j(cpt.arctanh(bm.asarray([0., 0.5])))),
        np.arctanh([0., 0.5]), atol=1e-6,
    )
    # aliases point at their canonical implementations
    assert cpt.arcsinh is cpt.asinh
    assert cpt.arctanh is cpt.atanh


def test_pytorch_arcsin_is_asin_not_arcsinh():
    """No duplicate ``arcsin`` should shadow asin with arcsinh."""
    assert cpt.arcsin is cpt.asin
    np.testing.assert_allclose(
        np.asarray(_j(cpt.arcsin(bm.asarray([0., 0.5])))),
        np.arcsin([0., 0.5]), atol=1e-6,
    )


def test_numpy_arcsinh_arctanh_present():
    np.testing.assert_allclose(
        np.asarray(_j(cn.arcsinh(bm.asarray([0., 1.])))), np.arcsinh([0., 1.]), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(_j(cn.arctanh(bm.asarray([0., 0.5])))), np.arctanh([0., 0.5]), atol=1e-6)


# --- einops module still imports (dead _optimize_transformation removed) ----

def test_einops_module_imports():
    import brainpy.math.einops as eio
    assert hasattr(eio, 'ein_rearrange')
    assert hasattr(eio, 'ein_reduce')
    assert hasattr(eio, 'ein_repeat')
    assert hasattr(eio, 'ein_shape')
    # the dead helper flagged by the audit must be gone
    assert not hasattr(eio, '_optimize_transformation')


# ===========================================================================
# 2. Coverage tests
# ===========================================================================

# --- compat_numpy: creation funcs -------------------------------------------

def test_compat_numpy_creation():
    assert cn.zeros((2, 2)).shape == (2, 2)
    assert cn.ones((3,)).shape == (3,)
    assert cn.empty((2,)).shape == (2,)
    assert float(_j(cn.full((2,), 4.))[0]) == 4.
    assert cn.eye(3).shape == (3, 3)
    assert cn.identity(3).shape == (3, 3)
    assert cn.arange(5).shape == (5,)
    assert cn.linspace(0., 1., 5).shape == (5,)
    assert cn.logspace(0., 1., 5).shape == (5,)
    a = bm.asarray(jnp.arange(4.))
    assert cn.zeros_like(a).shape == (4,)
    assert cn.ones_like(a).shape == (4,)
    assert cn.empty_like(a).shape == (4,)
    assert cn.full_like(a, 2.).shape == (4,)


def test_compat_numpy_linspace_retstep():
    r, step = cn.linspace(0., 1., 5, retstep=True)
    assert isinstance(r, Array)
    assert float(step) == pytest.approx(0.25)


def test_compat_numpy_array_and_asarray_with_arrays():
    res = cn.array([bm.asarray([1., 2.]), bm.asarray([3., 4.])])
    assert res.shape == (2, 2)
    res2 = cn.asarray([bm.asarray([1., 2.]), bm.asarray([3., 4.])])
    assert res2.shape == (2, 2)
    assert cn.asanyarray([1, 2, 3]).shape == (3,)
    assert cn.ascontiguousarray([1, 2, 3]).shape == (3,)


def test_compat_numpy_diag_tri_family():
    m = bm.asarray(jnp.arange(9.).reshape(3, 3))
    assert cn.diag(m).shape == (3,)
    assert cn.tril(m).shape == (3, 3)
    assert cn.triu(m).shape == (3, 3)
    assert cn.tri(3).shape == (3, 3)
    assert cn.diagonal(m).shape == (3,)
    assert cn.diagflat(bm.asarray([1., 2., 3.])).shape == (3, 3)


def test_compat_numpy_ufuncs():
    a = bm.asarray([1., 2., 3.])
    b = bm.asarray([4., 5., 6.])
    funcs_binary = [cn.add, cn.subtract, cn.multiply, cn.divide, cn.power,
                    cn.true_divide, cn.maximum, cn.minimum, cn.fmax, cn.fmin,
                    cn.hypot, cn.logaddexp, cn.logaddexp2, cn.copysign,
                    cn.nextafter, cn.remainder, cn.mod, cn.fmod, cn.float_power]
    for f in funcs_binary:
        assert _j(f(a, b)).shape == (3,)
    funcs_unary = [cn.negative, cn.positive, cn.reciprocal, cn.abs, cn.absolute,
                   cn.fabs, cn.exp, cn.exp2, cn.expm1, cn.log1p, cn.sqrt, cn.cbrt,
                   cn.square, cn.sign, cn.sin, cn.cos, cn.tan, cn.sinh, cn.cosh,
                   cn.tanh, cn.arcsin, cn.arccos, cn.arctan, cn.arcsinh, cn.arctanh,
                   cn.sinc, cn.deg2rad, cn.rad2deg, cn.degrees, cn.radians,
                   cn.round, cn.rint, cn.floor, cn.ceil, cn.trunc, cn.isfinite,
                   cn.isinf, cn.isnan, cn.signbit, cn.conj, cn.conjugate, cn.real,
                   cn.imag, cn.angle, cn.nan_to_num]
    pos = bm.asarray([0.1, 0.2, 0.3])
    for f in funcs_unary:
        assert _j(f(pos)).shape == (3,)
    # log family wants strictly positive
    for f in [cn.log, cn.log10, cn.log2]:
        assert _j(f(a)).shape == (3,)
    # gcd/lcm want integers
    ia, ib = bm.asarray([4, 6, 8]), bm.asarray([6, 9, 12])
    assert _j(cn.gcd(ia, ib)).shape == (3,)
    assert _j(cn.lcm(ia, ib)).shape == (3,)
    assert _j(cn.arctan2(a, b)).shape == (3,)
    assert _j(cn.heaviside(a, b)).shape == (3,)
    assert len(cn.frexp(a)) == 2
    assert len(cn.modf(a)) == 2


def test_compat_numpy_reductions():
    a = bm.asarray([[1., 2., 3.], [4., 5., 6.]])
    assert float(_j(cn.sum(a))) == 21.
    assert float(_j(cn.prod(bm.asarray([1., 2., 3., 4.])))) == 24.
    assert float(_j(cn.mean(a))) == pytest.approx(3.5)
    assert float(_j(cn.max(a))) == 6.
    assert float(_j(cn.min(a))) == 1.
    assert float(_j(cn.amax(a))) == 6.
    assert float(_j(cn.amin(a))) == 1.
    assert _j(cn.std(a)).shape == ()
    assert _j(cn.var(a)).shape == ()
    assert _j(cn.median(a)).shape == ()
    assert _j(cn.average(a)).shape == ()
    assert _j(cn.cumsum(a)).shape == (6,)
    assert _j(cn.cumprod(a)).shape == (6,)
    assert _j(cn.nansum(a)).shape == ()
    assert _j(cn.nanprod(a)).shape == ()
    assert _j(cn.nanmean(a)).shape == ()
    assert _j(cn.nanstd(a)).shape == ()
    assert _j(cn.nanvar(a)).shape == ()
    assert _j(cn.nanmedian(a)).shape == ()
    assert int(cn.argmax(a)) == 5
    assert int(cn.argmin(a)) == 0
    assert _j(cn.ptp(a)).shape == ()
    assert _j(cn.diff(a)).shape == (2, 2)
    assert _j(cn.nancumsum(a)).shape == (6,)
    assert _j(cn.nancumprod(a)).shape == (6,)
    assert int(cn.count_nonzero(a)) == 6
    assert _j(cn.percentile(a, 50)).shape == ()
    assert _j(cn.quantile(a, 0.5)).shape == ()


def test_compat_numpy_logic():
    a = bm.asarray([1., 2., 3.])
    b = bm.asarray([1., 0., 3.])
    assert _j(cn.equal(a, b)).shape == (3,)
    assert _j(cn.not_equal(a, b)).shape == (3,)
    assert _j(cn.greater(a, b)).shape == (3,)
    assert _j(cn.greater_equal(a, b)).shape == (3,)
    assert _j(cn.less(a, b)).shape == (3,)
    assert _j(cn.less_equal(a, b)).shape == (3,)
    assert bool(cn.array_equal(a, a))
    assert _j(cn.isclose(a, b)).shape == (3,)
    assert bool(cn.allclose(a, a))
    ba = bm.asarray([True, False, True])
    bb = bm.asarray([True, True, False])
    assert _j(cn.logical_and(ba, bb)).shape == (3,)
    assert _j(cn.logical_or(ba, bb)).shape == (3,)
    assert _j(cn.logical_xor(ba, bb)).shape == (3,)
    assert _j(cn.logical_not(ba)).shape == (3,)
    assert bool(cn.all(ba)) is False
    assert bool(cn.any(ba)) is True
    assert bool(cn.alltrue(ba)) is False
    assert bool(cn.sometrue(ba)) is True


def test_compat_numpy_bit_ops():
    a = bm.asarray([1, 2, 3])
    b = bm.asarray([3, 2, 1])
    assert _j(cn.bitwise_and(a, b)).shape == (3,)
    assert _j(cn.bitwise_or(a, b)).shape == (3,)
    assert _j(cn.bitwise_xor(a, b)).shape == (3,)
    assert _j(cn.bitwise_not(a)).shape == (3,)
    assert _j(cn.invert(a)).shape == (3,)
    assert _j(cn.left_shift(a, b)).shape == (3,)
    assert _j(cn.right_shift(a, b)).shape == (3,)


def test_compat_numpy_manipulation():
    a = bm.asarray(jnp.arange(6.))
    m = bm.asarray(jnp.arange(6.).reshape(2, 3))
    assert _j(cn.reshape(a, (2, 3))).shape == (2, 3)
    assert _j(cn.ravel(m)).shape == (6,)
    assert _j(cn.moveaxis(m, 0, 1)).shape == (3, 2)
    assert _j(cn.transpose(m)).shape == (3, 2)
    assert _j(cn.swapaxes(m, 0, 1)).shape == (3, 2)
    assert _j(cn.concatenate([a, a])).shape == (12,)
    assert _j(cn.stack([a, a])).shape == (2, 6)
    assert _j(cn.vstack([a, a])).shape == (2, 6)
    assert _j(cn.hstack([a, a])).shape == (12,)
    assert _j(cn.dstack([a, a])).shape == (1, 6, 2)
    assert _j(cn.column_stack([a, a])).shape == (6, 2)
    assert len(cn.split(a, 2)) == 2
    assert len(cn.array_split(a, 4)) == 4
    assert _j(cn.tile(a, 2)).shape == (12,)
    assert _j(cn.repeat(a, 2)).shape == (12,)
    assert _j(cn.flip(a)).shape == (6,)
    assert _j(cn.fliplr(m)).shape == (2, 3)
    assert _j(cn.flipud(m)).shape == (2, 3)
    assert _j(cn.roll(a, 1)).shape == (6,)
    assert _j(cn.atleast_1d(bm.asarray(1.))).shape == (1,)
    assert _j(cn.atleast_2d(a)).shape == (1, 6)
    assert _j(cn.atleast_3d(a)).shape == (1, 6, 1)
    assert _j(cn.expand_dims(a, 0)).shape == (1, 6)
    assert _j(cn.squeeze(bm.asarray(jnp.ones((1, 3))))).shape == (3,)
    assert _j(cn.append(a, a)).shape == (12,)
    assert _j(cn.sort(a)).shape == (6,)
    assert _j(cn.argsort(a)).shape == (6,)
    assert _j(cn.unique(bm.asarray([1, 1, 2, 3]))).shape == (3,)
    assert _j(cn.row_stack([a, a])).shape == (2, 6)


def test_compat_numpy_indexing_and_where():
    a = bm.asarray(jnp.arange(6.))
    assert _j(cn.where(a > 2, a, 0.)).shape == (6,)
    assert len(cn.nonzero(a)) == 1
    assert _j(cn.argwhere(a > 2).flatten()).shape[0] >= 0
    assert _j(cn.flatnonzero(a)).shape[0] == 5
    assert int(cn.searchsorted(a, 3.5)) == 4
    assert _j(cn.take(a, bm.asarray([0, 2, 4]))).shape == (3,)
    assert _j(cn.select([a > 3], [a])).shape == (6,)
    assert _j(cn.extract(a > 3, a)).shape[0] == 2
    assert len(cn.tril_indices(3)) == 2
    assert len(cn.triu_indices(3)) == 2
    m = bm.asarray(jnp.ones((3, 3)))
    assert len(cn.tril_indices_from(m)) == 2
    assert len(cn.triu_indices_from(m)) == 2


def test_compat_numpy_linalg():
    a = bm.asarray([1., 2., 3.])
    b = bm.asarray([4., 5., 6.])
    m = bm.asarray(jnp.arange(9.).reshape(3, 3))
    assert _j(cn.dot(a, b)).shape == ()
    assert _j(cn.vdot(a, b)).shape == ()
    assert _j(cn.inner(a, b)).shape == ()
    assert _j(cn.outer(a, b)).shape == (3, 3)
    assert _j(cn.kron(a, b)).shape == (9,)
    assert _j(cn.matmul(m, m)).shape == (3, 3)
    assert _j(cn.trace(m)).shape == ()
    assert _j(cn.tensordot(m, m)).shape == ()


def test_compat_numpy_misc_helpers():
    a = bm.asarray([1., 2., 3.])
    assert cn.shape(a) == (3,)
    assert cn.shape([[1, 2]]) == (1, 2)
    assert cn.shape(0) == ()
    assert cn.size(a) == 3
    assert cn.size(bm.asarray(jnp.ones((2, 3))), 1) == 3
    assert cn.size([1, 2, 3]) == 3
    assert int(cn.ndim(a)) == 1
    assert float(cn.asscalar(bm.asarray(7.))) == 7.
    assert cn.matrix([[1, 2], [3, 4]]).shape == (2, 2)
    assert cn.asmatrix([1, 2, 3]).shape == (1, 3)
    assert cn.mat([1, 2, 3]).shape == (1, 3)
    assert cn.msort(bm.asarray(jnp.array([[3., 1.], [2., 4.]]))).shape == (2, 2)
    assert cn.common_type(jnp.array([1., 2.])) is not None
    assert cn.common_type(jnp.array([1 + 1j])) is not None  # complex branch
    assert _j(cn.frombuffer(b'\x01\x02\x03\x04', dtype=np.int8)).shape == (4,)
    assert _j(cn.meshgrid(a, a))[0].shape == (3, 3)
    assert _j(cn.broadcast_to(a, (2, 3))).shape == (2, 3)
    assert cn.broadcast_shapes((3,), (2, 3)) == (2, 3)
    assert _j(cn.pad(a, 1)).shape == (5,)
    assert _j(cn.clip(a, 1.5, 2.5)).shape == (3,)
    assert _j(cn.interp(bm.asarray([1.5]), a, a)).shape == (1,)
    assert _j(cn.einsum('i,i->', a, a)).shape == ()
    assert _j(cn.gradient(a)).shape == (3,)
    assert _j(cn.histogram(a)[0]).shape == (10,)
    assert _j(cn.bincount(bm.asarray([0, 1, 1, 2]))).shape == (3,)
    assert _j(cn.digitize(a, bm.asarray([0., 2.]))).shape == (3,)


def test_compat_numpy_window_and_constants():
    assert _j(cn.bartlett(4)).shape == (4,)
    assert _j(cn.blackman(4)).shape == (4,)
    assert _j(cn.hamming(4)).shape == (4,)
    assert _j(cn.hanning(4)).shape == (4,)
    assert _j(cn.kaiser(4, 1.0)).shape == (4,)
    assert cn.e == pytest.approx(np.e)
    assert cn.pi == pytest.approx(np.pi)
    assert np.isinf(cn.inf)


def test_compat_numpy_inplace_helpers_and_errors():
    a = bm.asarray(jnp.arange(6))
    cn.place(a, jnp.array([True, False] * 3), [10, 20, 30])
    b = bm.asarray(jnp.arange(6))
    cn.put(b, jnp.array([0, 1]), jnp.array([9, 8]))
    assert int(_j(b)[0]) == 9
    c = bm.asarray(jnp.zeros(3))
    cn.copyto(c, jnp.ones(3))
    assert float(_j(c)[0]) == 1.
    # error paths (non-Array inputs)
    with pytest.raises(ValueError):
        cn.place(jnp.arange(6), jnp.array([True] * 6), [1])
    with pytest.raises(ValueError):
        cn.put(jnp.arange(6), [0], [9])
    with pytest.raises(ValueError):
        cn.putmask(jnp.arange(6), jnp.arange(6) > 2, jnp.arange(6))
    with pytest.raises(ValueError):
        cn.copyto(jnp.zeros(3), jnp.ones(3))


def test_compat_numpy_in1d_and_set_ops():
    a = bm.asarray([1, 2, 3, 4])
    b = bm.asarray([2, 4])
    assert _j(cn.in1d(a, b)).shape == (4,)
    assert _j(cn.in1d(a, b, invert=True)).shape == (4,)
    assert _j(cn.intersect1d(a, b)).shape == (2,)
    assert _j(cn.union1d(a, b)).shape[0] == 4
    assert _j(cn.setdiff1d(a, b)).shape == (2,)
    assert _j(cn.isin(a, b)).shape == (4,)


def test_compat_numpy_dtype_helpers():
    assert cn.issubdtype(jnp.float32, jnp.floating)
    assert cn.can_cast(jnp.int32, jnp.int64)
    assert cn.result_type(jnp.int32, jnp.float32) is not None
    assert cn.promote_types(jnp.int32, jnp.float32) is not None
    assert cn.finfo(jnp.float32).bits == 32
    assert cn.iinfo(jnp.int32).bits == 32


# --- compat_pytorch ---------------------------------------------------------

def test_pytorch_shape_ops():
    a = bm.asarray(jnp.arange(24.).reshape(2, 3, 4))
    assert cpt.flatten(a).shape == (24,)
    assert cpt.flatten(a, start_dim=1).shape == (2, 12)
    assert cpt.flatten(a, start_dim=1, end_dim=2).shape == (2, 12)
    assert cpt.flatten(a, start_dim=-2).shape == (2, 12)
    assert cpt.flatten(bm.asarray(jnp.array(3.))).shape == (1,)
    assert _j(cpt.unflatten(bm.asarray(jnp.arange(6.)), 0, (2, 3))).shape == (2, 3)
    assert _j(cpt.unsqueeze(bm.asarray(jnp.arange(3.)), 0)).shape == (1, 3)
    assert _j(cpt.cat([bm.asarray([1., 2.]), bm.asarray([3., 4.])])).shape == (4,)


def test_pytorch_flatten_errors():
    a = bm.asarray(jnp.arange(6.).reshape(2, 3))
    with pytest.raises(ValueError):
        cpt.flatten(a, start_dim=5)
    with pytest.raises(ValueError):
        cpt.flatten(a, end_dim=5)


def test_pytorch_math_ops_no_out():
    a = bm.asarray([0.1, 0.2, 0.3])
    for f in [cpt.abs, cpt.absolute, cpt.acos, cpt.arccos, cpt.acosh, cpt.arccosh,
              cpt.asin, cpt.arcsin, cpt.asinh, cpt.arcsinh, cpt.atan, cpt.arctan,
              cpt.atanh, cpt.arctanh]:
        # acosh needs x >= 1
        x = bm.asarray([1.1, 1.2, 1.3]) if f in (cpt.acosh, cpt.arccosh) else a
        assert _j(f(x)).shape == (3,)
    assert _j(cpt.angle(bm.asarray([1 + 1j, 2 - 1j]))).shape == (2,)
    assert _j(cpt.atan2(a, a)).shape == (3,)
    assert _j(cpt.arctan2(a, a)).shape == (3,)


def test_pytorch_add_family():
    a = bm.asarray([1., 2., 3.])
    b = bm.asarray([4., 5., 6.])
    np.testing.assert_allclose(np.asarray(_j(cpt.add(a, b))), [5., 7., 9.])
    np.testing.assert_allclose(np.asarray(_j(cpt.add(a, b, alpha=2))), [9., 12., 15.])
    assert _j(cpt.addcdiv(a, b, b, value=2)).shape == (3,)
    assert _j(cpt.addcmul(a, b, b, value=2)).shape == (3,)


def test_pytorch_out_paths():
    a = bm.asarray([-1., -2., -3.])
    out = bm.zeros((3,))
    r = cpt.abs(a, out=out)
    assert r is out
    np.testing.assert_allclose(np.asarray(_j(out)), [1., 2., 3.])
    out2 = bm.zeros((3,))
    r2 = cpt.addcdiv(bm.asarray([1., 1., 1.]), bm.asarray([2., 2., 2.]),
                     bm.asarray([2., 2., 2.]), value=1, out=out2)
    assert r2 is out2


def test_pytorch_unary_out_paths():
    """Exercise the ``out=`` branch of every unary pytorch math op (H-14)."""
    a = bm.asarray([0.2, 0.3, 0.4])
    high = bm.asarray([1.1, 1.2, 1.3])
    cases = [
        (cpt.acos, a), (cpt.arccos, a), (cpt.acosh, high), (cpt.arccosh, high),
        (cpt.asin, a), (cpt.arcsin, a), (cpt.asinh, a), (cpt.arcsinh, a),
        (cpt.atan, a), (cpt.arctan, a), (cpt.atanh, a), (cpt.arctanh, a),
        (cpt.absolute, a),
    ]
    for f, x in cases:
        out = bm.zeros((3,))
        r = f(x, out=out)
        assert r is out
        assert _finite(out)
    # binary / complex out= paths
    out = bm.zeros((3,))
    assert cpt.atan2(a, a, out=out) is out
    out = bm.zeros((3,))
    assert cpt.arctan2(a, a, out=out) is out
    cout = bm.zeros((3,))
    assert cpt.angle(bm.asarray([1 + 1j, 2 + 0j, 0 + 1j]), out=cout) is cout


def test_pytorch_flatten_negative_end_dim_error():
    a = bm.asarray(jnp.arange(6.).reshape(2, 3))
    with pytest.raises(ValueError):
        cpt.flatten(a, end_dim=-10)


def test_pytorch_clamp_aliases():
    a = bm.asarray([1., 5., 9.])
    np.testing.assert_allclose(np.asarray(_j(cpt.clamp_max(a, 4.))), [1., 4., 4.])
    np.testing.assert_allclose(np.asarray(_j(cpt.clamp_min(a, 4.))), [4., 5., 9.])


def test_pytorch_tensor_alias():
    assert cpt.Tensor is Array


# --- compat_tensorflow ------------------------------------------------------

def test_tensorflow_reductions():
    a = bm.asarray([[1., 2., 3.], [4., 5., 6.]])
    assert float(_j(ctf.reduce_sum(a))) == 21.
    assert float(_j(ctf.reduce_max(a))) == 6.
    assert float(_j(ctf.reduce_min(a))) == 1.
    assert float(_j(ctf.reduce_mean(a))) == pytest.approx(3.5)
    assert float(_j(ctf.reduce_prod(a))) == 720.
    assert _j(ctf.reduce_std(a)).shape == ()
    assert _j(ctf.reduce_variance(a)).shape == ()
    assert _j(ctf.reduce_euclidean_norm(a)).shape == ()
    ba = bm.asarray([[True, False], [True, True]])
    assert bool(ctf.reduce_all(ba)) is False
    assert bool(ctf.reduce_any(ba)) is True
    # axis variants
    assert _j(ctf.reduce_max(a, axis=1)).shape == (2,)
    assert _j(ctf.reduce_sum(a, axis=0, keepdims=True)).shape == (1, 3)
    assert _j(ctf.reduce_euclidean_norm(a, axis=1)).shape == (2,)


def test_tensorflow_segment_ops():
    data = bm.asarray([1., 2., 3., 4.])
    seg = bm.asarray([0, 0, 1, 1])
    assert _j(ctf.segment_sum(data, seg)).shape == (2,)
    assert _j(ctf.segment_prod(data, seg)).shape == (2,)
    assert _j(ctf.segment_max(data, seg)).shape == (2,)
    assert _j(ctf.segment_min(data, seg)).shape == (2,)
    np.testing.assert_allclose(np.asarray(_j(ctf.segment_mean(data, seg))), [1.5, 3.5])
    assert _j(ctf.unsorted_segment_sum(data, seg, 2)).shape == (2,)
    assert _j(ctf.unsorted_segment_prod(data, seg, 2)).shape == (2,)
    assert _j(ctf.unsorted_segment_max(data, seg, 2)).shape == (2,)
    assert _j(ctf.unsorted_segment_min(data, seg, 2)).shape == (2,)
    assert _j(ctf.unsorted_segment_mean(data, seg, 2)).shape == (2,)
    assert _j(ctf.unsorted_segment_sqrt_n(data, seg, 2)).shape == (2,)


def test_tensorflow_cast_clip_concat():
    a = bm.asarray([1.4, 2.6, 3.1])
    casted = ctf.cast(a, jnp.int32)
    assert casted.dtype == jnp.int32
    np.testing.assert_allclose(np.asarray(_j(ctf.clip_by_value(a, 2., 3.))), [2., 2.6, 3.])
    assert _j(ctf.concat([a, a])).shape == (6,)


# --- activations ------------------------------------------------------------

def test_activations_basic():
    x = bm.asarray([-2., -0.5, 0., 0.5, 2.])
    for f in [act.relu, act.relu6, act.sigmoid, act.softplus, act.silu, act.swish,
              act.mish, act.selu, act.elu, act.celu, act.soft_sign, act.log_sigmoid,
              act.hard_sigmoid, act.hard_silu, act.hard_swish, act.tanh_shrink,
              act.leaky_relu, act.hard_shrink, act.soft_shrink, act.prelu,
              act.identity]:
        r = f(x)
        assert np.asarray(_j(r)).shape == (5,)
        assert _finite(r)


def test_activations_relu_correct():
    x = bm.asarray([-1., 0., 2.])
    np.testing.assert_allclose(np.asarray(_j(act.relu(x))), [0., 0., 2.])
    np.testing.assert_allclose(np.asarray(_j(act.relu6(bm.asarray([-1., 3., 9.])))), [0., 3., 6.])


def test_activations_gelu_both():
    x = bm.asarray([-1., 0., 1., 2.])
    assert _finite(act.gelu(x, approximate=True))
    assert _finite(act.gelu(x, approximate=False))


def test_activations_softmax_family():
    x = bm.asarray([1., 2., 3.])
    sm = act.softmax(x)
    assert float(jnp.sum(_j(sm))) == pytest.approx(1.0, abs=1e-6)
    ls = act.log_softmax(x)
    np.testing.assert_allclose(np.asarray(_j(jnp.exp(ls))), np.asarray(_j(sm)), atol=1e-6)
    smn = act.softmin(x)
    assert float(jnp.sum(_j(smn))) == pytest.approx(1.0, abs=1e-6)
    assert act.soft_max is act.softmax


def test_activations_softmax_large_inputs_stable():
    x = bm.asarray([1000., 1001., 1002.])
    assert _finite(act.softmax(x))
    assert _finite(act.log_softmax(x))
    assert _finite(act.softmin(x))


def test_activations_tanh_and_glu():
    x = bm.asarray([-1., 0., 1.])
    np.testing.assert_allclose(np.asarray(_j(act.tanh(x))), np.tanh([-1., 0., 1.]), atol=1e-6)
    assert _j(act.glu(bm.asarray(jnp.arange(4.)))).shape == (2,)
    with pytest.raises(AssertionError):
        act.glu(bm.asarray(jnp.arange(3.)))  # odd axis size


def test_activations_one_hot_and_normalize():
    oh = act.one_hot(bm.asarray([0, 1, 2]), 3)
    assert _j(oh).shape == (3, 3)
    np.testing.assert_allclose(np.asarray(_j(oh)), np.eye(3))
    # out-of-range indices -> all zeros
    oh2 = act.one_hot(bm.asarray([-1, 5]), 3)
    np.testing.assert_allclose(np.asarray(_j(oh2)), np.zeros((2, 3)))
    n = act.normalize(bm.asarray([1., 2., 3., 4.]))
    assert _finite(n)
    assert float(jnp.mean(_j(n))) == pytest.approx(0.0, abs=1e-5)


def test_activations_rrelu():
    r = act.rrelu(bm.asarray([-1., -2., 1., 2.]))
    assert _j(r).shape == (4,)
    # positive entries pass through unchanged
    assert float(_j(r)[2]) == 1.
    assert float(_j(r)[3]) == 2.


def test_activations_get_dispatch():
    assert act.get('relu') is act.relu
    assert act.get(None) is None
    fn = (lambda x: x)
    assert act.get(fn) is fn
    with pytest.raises(ValueError):
        act.get('this_is_not_an_activation')
    with pytest.raises(ValueError):
        act.get(123)


def test_activations_accept_plain_jax_arrays():
    x = jnp.array([-1., 0., 1.])
    for f in [act.relu, act.sigmoid, act.softmax, act.softmin, act.elu, act.gelu]:
        assert _finite(f(x))


def test_activations_axis_out_of_bounds():
    # one_hot routes axis through _canonicalize_axis -> ValueError on OOB
    with pytest.raises(ValueError):
        act.one_hot(jnp.array([0, 1, 2]), 3, axis=5)
    # log_softmax/softmax over a non-existent axis raises
    with pytest.raises(Exception):
        act.log_softmax(jnp.arange(6.), axis=5)


def test_activations_one_hot_axis_and_dtype():
    oh = act.one_hot(jnp.array([0, 1, 2]), 3, axis=0)
    assert _j(oh).shape == (3, 3)
    oh_i = act.one_hot(jnp.array([0, 1]), 2, dtype=jnp.int32)
    assert _j(oh_i).dtype == jnp.int32


def test_activations_hard_tanh_clamping():
    x = bm.asarray([-2., -0.5, 0.5, 2.])
    np.testing.assert_allclose(np.asarray(_j(act.hard_tanh(x))), [-1., -0.5, 0.5, 1.])


def test_activations_softplus_threshold_branch():
    # values above threshold revert to the linear branch
    x = bm.asarray([0.5, 25.0, 50.0])
    r = act.softplus(x, beta=1., threshold=20.)
    assert _finite(r)
    assert float(_j(r)[1]) == pytest.approx(25.0, rel=1e-5)


# --- others -----------------------------------------------------------------

def test_others_shared_args_over_time():
    r = bo.shared_args_over_time(num_step=5)
    assert r['i'].shape == (5,)
    assert r['t'].shape == (5,)
    assert r['dt'].shape == (5,)
    r2 = bo.shared_args_over_time(duration=1.0, dt=0.1, include_dt=False)
    assert r2['i'].shape == (10,)
    assert 'dt' not in r2


def test_others_clip_by_norm():
    t = jnp.array([3., 4.])
    r = bo.clip_by_norm(t, 1.0)
    assert float(jnp.linalg.norm(_j(r))) <= 1.0 + 1e-5
    # pytree input
    r2 = bo.clip_by_norm({'a': jnp.array([3., 4.])}, 1.0)
    assert 'a' in r2


def test_others_exprel():
    r = bo.exprel(jnp.array([0., 1., -1.]))
    assert _finite(r)
    # exprel(0) == 1 (removable singularity)
    assert float(_j(r)[0]) == pytest.approx(1.0, abs=1e-4)
    # exprel(x) == (exp(x)-1)/x away from zero
    assert float(_j(r)[1]) == pytest.approx((np.e - 1.0), rel=1e-3)
    # float64 threshold branch
    r64 = bo.exprel(jnp.array([0.5]))
    assert _finite(r64)


def test_others_is_float_type():
    assert bo.is_float_type(jnp.array([1., 2.]))
    assert not bo.is_float_type(jnp.array([1, 2]))


def test_others_add_axis_axes():
    x = jnp.arange(3.)
    assert _j(bo.add_axis(x, 0)).shape == (1, 3)
    r = bo.add_axes(x, n_axes=2, pos2len={0: 4})
    assert _j(r).shape == (4, 3)


# --- _utils -----------------------------------------------------------------

def test_utils_as_jax_array_and_is_leaf():
    a = bm.asarray([1., 2., 3.])
    assert isinstance(butils._as_jax_array_(a), jax.Array)
    assert butils._as_jax_array_(5) == 5
    assert butils._is_leaf(a) is True
    assert butils._is_leaf(jnp.array([1.])) is False


def test_utils_compatible_wrapper_kwargs_translation():
    a = bm.asarray([[1., 2.], [3., 4.]])
    # PyTorch dim -> axis
    np.testing.assert_allclose(
        np.asarray(_j(bm.sum(a, dim=0))), [4., 6.])
    # PyTorch keepdim -> keepdims
    assert _j(bm.sum(a, axis=0, keepdim=True)).shape == (1, 2)
    # TensorFlow keep_dims -> keepdims
    assert _j(bm.sum(a, axis=0, keep_dims=True)).shape == (1, 2)


def test_utils_wrapper_returns_brainpy_array():
    r = bm.add(bm.asarray([1., 2.]), bm.asarray([3., 4.]))
    assert isinstance(r, Array)


def test_utils_wrapper_doc_and_name():
    # _compatible_with_brainpy_array preserves the wrapped function name
    assert cn.sum.__name__ == 'sum'
    assert 'brainpy Array/Variable' in cn.sum.__doc__


# --- einops -----------------------------------------------------------------

def test_einops_rearrange():
    x = jnp.arange(24.).reshape(2, 3, 4)
    assert bein.ein_rearrange(x, 'a b c -> a c b').shape == (2, 4, 3)
    assert bein.ein_rearrange(x, 'a b c -> (a b) c').shape == (6, 4)
    assert bein.ein_rearrange(x, 'a b c -> a b c').shape == (2, 3, 4)
    # split an axis
    assert bein.ein_rearrange(jnp.arange(12.), '(a b) -> a b', a=3).shape == (3, 4)


def test_einops_reduce():
    x = jnp.arange(24.).reshape(2, 3, 4)
    assert bein.ein_reduce(x, 'a b c -> a c', 'mean').shape == (2, 4)
    assert bein.ein_reduce(x, 'a b c -> a', 'sum').shape == (2,)
    assert bein.ein_reduce(x, 'a b c -> b c', 'max').shape == (3, 4)
    assert bein.ein_reduce(x, 'a b c -> b c', 'min').shape == (3, 4)
    assert bein.ein_reduce(x, 'a b c -> b c', 'prod').shape == (3, 4)
    # pooling-style reduce with explicit axis lengths
    y = jnp.arange(2 * 2 * 4 * 4.).reshape(2, 2, 4, 4)
    assert bein.ein_reduce(y, 'b c (h h2) (w w2) -> b c h w', 'max', h2=2, w2=2).shape == (2, 2, 2, 2)


def test_einops_reduce_any_all():
    b = jnp.array([[True, False, True], [False, False, True]])
    assert bein.ein_reduce(b, 'a c -> c', 'any').shape == (3,)
    assert bein.ein_reduce(b, 'a c -> c', 'all').shape == (3,)
    np.testing.assert_array_equal(
        np.asarray(_j(bein.ein_reduce(b, 'a c -> c', 'any'))), [True, False, True])
    np.testing.assert_array_equal(
        np.asarray(_j(bein.ein_reduce(b, 'a c -> c', 'all'))), [False, False, True])


def test_einops_repeat():
    img = jnp.arange(6.).reshape(2, 3)
    assert bein.ein_repeat(img, 'h w -> h w c', c=4).shape == (2, 3, 4)
    assert bein.ein_repeat(img, 'h w -> (h2 h) w', h2=2).shape == (4, 3)


def test_einops_shape():
    x = jnp.zeros((2, 3, 5, 7))
    assert bein.ein_shape(x, 'batch _ h w') == {'batch': 2, 'h': 5, 'w': 7}
    assert bein.ein_shape(x, 'a b c d') == {'a': 2, 'b': 3, 'c': 5, 'd': 7}


def test_einops_reduce_callable_reduction():
    x = jnp.arange(24.).reshape(2, 3, 4)
    out = bein.ein_reduce(x, 'a b c -> a c', lambda t, axes: t.sum(axis=axes))
    assert out.shape == (2, 4)


def test_einops_mean_requires_float():
    x = jnp.arange(24).reshape(2, 3, 4)  # integer tensor
    with pytest.raises(Exception):
        bein.ein_reduce(x, 'a b c -> a c', 'mean')


def test_einops_error_message_wrapped():
    from brainpy.math.einops_parsing import EinopsError
    with pytest.raises(EinopsError):
        bein.ein_rearrange(jnp.arange(6.), 'a b c -> a b c')  # wrong ndim


def test_einops_enumerate_directions_internal():
    x = jnp.zeros((2, 3))
    dirs = bein._enumerate_directions(x)
    assert len(dirs) == 2
    assert _j(dirs[0]).shape == (2, 1)
    assert _j(dirs[1]).shape == (1, 3)


def test_einops_ellipsis_patterns():
    x = jnp.arange(24.).reshape(2, 3, 4)
    # reduce trailing axis, keep ellipsis dims
    assert bein.ein_reduce(x, '... c -> ...', 'sum').shape == (2, 3)
    # move leading axis to the end across an ellipsis
    assert bein.ein_rearrange(x, 'a ... -> ... a').shape == (3, 4, 2)
    # repeat with ellipsis
    assert bein.ein_repeat(jnp.arange(6.).reshape(2, 3), '... -> ... r', r=2).shape == (2, 3, 2)


def test_einops_shape_with_ellipsis():
    x = jnp.zeros((2, 3, 5, 7))
    assert bein.ein_shape(x, 'b ... w') == {'b': 2, 'w': 7}


def test_einops_error_branches():
    from brainpy.math.einops_parsing import EinopsError
    x = jnp.arange(24.).reshape(2, 3, 4)
    # identifiers only on one side of a rearrange
    with pytest.raises(EinopsError):
        bein.ein_rearrange(x, 'a b c -> a b')
    # repeat without a size for a new axis
    with pytest.raises(EinopsError):
        bein.ein_repeat(jnp.arange(6.).reshape(2, 3), 'h w -> h w c')
    # extra identifier on the right of a reduce
    with pytest.raises(EinopsError):
        bein.ein_reduce(x, 'a b c -> a b c d', 'sum')
    # unknown reduction name
    with pytest.raises(EinopsError):
        bein.ein_reduce(x, 'a b c -> a', 'median')
    # composed axes can't be parsed by ein_shape
    with pytest.raises(RuntimeError):
        bein.ein_shape(jnp.zeros((6,)), '(a b)')


def test_einops_list_input_passthrough_identity():
    # NOTE: the docstrings advertise stacking list-of-tensors input, but this
    # port does not stack the list -- an identity pattern is a no-op and returns
    # the list unchanged. Pin the current (documented-but-incomplete) behaviour.
    imgs = [jnp.zeros((3, 4)) for _ in range(5)]
    out = bein.ein_rearrange(imgs, 'b h w -> b h w')
    assert isinstance(out, list)
    assert len(out) == 5
