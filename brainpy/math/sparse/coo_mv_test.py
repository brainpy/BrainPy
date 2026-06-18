# -*- coding: utf-8 -*-
"""Regression tests for ``brainpy/math/sparse/coo_mv.py``.

``coomv`` converts COO indices to CSR (``brainevent.coo2csr``) before delegating
to ``brainevent.CSR``. These tests check both orientations and the scalar-weight
broadcast path against a dense numpy reference, with unsorted COO triples (so the
``coo2csr`` permutation of ``data`` is exercised).
"""

import jax
import jax.numpy as jnp
import numpy as np

from brainpy.math.sparse.coo_mv import coomv


# Deliberately UNSORTED COO triples for a 3 x 4 matrix:
#   [[0, 2, 0, 4],
#    [1, 0, 0, 0],
#    [0, 3, 0, 2]]
_ROWS = np.array([2, 0, 1, 0, 2])
_COLS = np.array([1, 1, 0, 3, 3])
_VALS = np.array([3., 2., 1., 4., 2.])
_SHAPE = (3, 4)


def _dense():
    m = np.zeros(_SHAPE, dtype=np.float32)
    for v, r, c in zip(_VALS, _ROWS, _COLS):
        m[r, c] = v
    return m


def test_coomv_no_transpose_matches_dense():
    v = jnp.arange(4, dtype=jnp.float32)
    out = np.asarray(coomv(_VALS, _ROWS, _COLS, v, shape=_SHAPE, transpose=False))
    assert out.shape == (3,)
    np.testing.assert_allclose(out, _dense() @ np.asarray(v), rtol=1e-5, atol=1e-5)


def test_coomv_transpose_matches_dense():
    v = jnp.arange(3, dtype=jnp.float32)
    out = np.asarray(coomv(_VALS, _ROWS, _COLS, v, shape=_SHAPE, transpose=True))
    assert out.shape == (4,)
    np.testing.assert_allclose(out, _dense().T @ np.asarray(v), rtol=1e-5, atol=1e-5)


def test_coomv_scalar_weight_broadcast():
    # scalar weight -> every stored entry uses the same value.
    v = jnp.arange(4, dtype=jnp.float32)
    out = np.asarray(coomv(2.0, _ROWS, _COLS, v, shape=_SHAPE, transpose=False))
    ref = np.zeros(_SHAPE, dtype=np.float32)
    ref[_ROWS, _COLS] = 2.0
    np.testing.assert_allclose(out, ref @ np.asarray(v), rtol=1e-5, atol=1e-5)


def test_coomv_grad_scalar_weight():
    v = jnp.arange(4, dtype=jnp.float32)

    def f(s):
        return coomv(s, _ROWS, _COLS, v, shape=_SHAPE, transpose=False).sum()

    g = float(jax.grad(f)(2.0))
    # d/ds sum(A(s) @ v) = sum over stored entries of v[col]
    np.testing.assert_allclose(g, float(jnp.asarray(v)[_COLS].sum()), rtol=1e-5, atol=1e-5)
