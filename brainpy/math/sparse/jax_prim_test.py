# -*- coding: utf-8 -*-
"""Coverage tests for ``brainpy/math/sparse/jax_prim.py``.

Exercises ``seg_matmul`` (the public ``segment_sum``-based sparse matmul) and its
two private helpers:

* ``_matmul_with_left_sparse``  (sparse @ dense) for 1-D and 2-D dense operands.
* ``_matmul_with_right_sparse`` (dense @ sparse) for 1-D and 2-D dense operands.
* the ``seg_matmul`` dispatcher and its two ValueError guards
  (both-sparse and both-dense).
* the ``len(shape) != 2`` ValueError inside both helpers.

All results are checked against a dense numpy reference built from the same COO
triples, with tiny matrices.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import brainpy.math as bm
from brainpy.math.sparse.jax_prim import seg_matmul


# A small known sparse matrix (3 x 4):
#   [[0, 2, 0, 4],
#    [1, 0, 0, 0],
#    [0, 3, 0, 2]]
_VALUES = [2, 4, 1, 3, 2]
_ROWS = [0, 0, 1, 2, 2]
_COLS = [1, 3, 0, 1, 3]
_SHAPE = (3, 4)


def _dense_ref():
    m = np.zeros(_SHAPE, dtype=np.float32)
    for v, r, c in zip(_VALUES, _ROWS, _COLS):
        m[r, c] = v
    return m


def _sparse():
    return {
        'data': bm.asarray(_VALUES, dtype=bm.float32),
        'index': (bm.asarray(_ROWS), bm.asarray(_COLS)),
        'shape': _SHAPE,
    }


# ---------------------------------------------------------------------------
# left sparse: sparse @ dense
# ---------------------------------------------------------------------------

def test_left_sparse_1d_dense():
    dense = _dense_ref()
    B = jnp.arange(4, dtype=jnp.float32)
    out = np.asarray(seg_matmul(_sparse(), B))
    assert out.shape == (3,)
    np.testing.assert_allclose(out, dense @ np.asarray(B), rtol=1e-5, atol=1e-5)


def test_left_sparse_2d_dense():
    dense = _dense_ref()
    B = jnp.arange(4 * 2, dtype=jnp.float32).reshape(4, 2)
    out = np.asarray(seg_matmul(_sparse(), B))
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out, dense @ np.asarray(B), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# right sparse: dense @ sparse
# ---------------------------------------------------------------------------

def test_right_sparse_1d_dense():
    dense = _dense_ref()
    A = jnp.arange(3, dtype=jnp.float32)
    out = np.asarray(seg_matmul(A, _sparse()))
    assert out.shape == (4,)
    np.testing.assert_allclose(out, np.asarray(A) @ dense, rtol=1e-5, atol=1e-5)


def test_right_sparse_2d_dense():
    dense = _dense_ref()
    A = jnp.arange(2 * 3, dtype=jnp.float32).reshape(2, 3)
    out = np.asarray(seg_matmul(A, _sparse()))
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out, np.asarray(A) @ dense, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# brainpy Array operands route through as_jax unwrap
# ---------------------------------------------------------------------------

def test_left_sparse_accepts_brainpy_array_dense():
    dense = _dense_ref()
    B = bm.asarray(jnp.arange(4, dtype=jnp.float32))
    out = np.asarray(seg_matmul(_sparse(), B))
    np.testing.assert_allclose(out, dense @ np.arange(4), rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# dispatcher error branches
# ---------------------------------------------------------------------------

def test_both_sparse_raises():
    with pytest.raises(ValueError):
        seg_matmul(_sparse(), _sparse())


def test_both_dense_raises():
    A = jnp.arange(3, dtype=jnp.float32)
    B = jnp.arange(4, dtype=jnp.float32)
    with pytest.raises(ValueError):
        seg_matmul(A, B)


# ---------------------------------------------------------------------------
# non-2D sparse shape error inside both helpers
# ---------------------------------------------------------------------------

def test_left_sparse_non_2d_shape_raises():
    bad = {
        'data': bm.asarray(_VALUES, dtype=bm.float32),
        'index': (bm.asarray(_ROWS), bm.asarray(_COLS)),
        'shape': (3, 4, 1),  # not 2-D
    }
    with pytest.raises(ValueError):
        seg_matmul(bad, jnp.arange(4, dtype=jnp.float32))


def test_right_sparse_non_2d_shape_raises():
    bad = {
        'data': bm.asarray(_VALUES, dtype=bm.float32),
        'index': (bm.asarray(_ROWS), bm.asarray(_COLS)),
        'shape': (3, 4, 1),  # not 2-D
    }
    with pytest.raises(ValueError):
        seg_matmul(jnp.arange(3, dtype=jnp.float32), bad)
