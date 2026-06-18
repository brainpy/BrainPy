# -*- coding: utf-8 -*-
"""Regression tests for ``brainpy/math/sparse/csr_mm.py``.

Guards the ``transpose=True`` branch of :func:`csrmm` (must compute ``Aᵀ @ B``,
not ``B @ A``) against a dense numpy reference, including its autodiff.
"""

import jax
import jax.numpy as jnp
import numpy as np

import brainevent

from brainpy.math.sparse.csr_mm import csrmm


# 3 x 4 sparse matrix:
#   [[0, 2, 0, 4],
#    [1, 0, 0, 0],
#    [0, 3, 0, 2]]
_ROWS = np.array([0, 0, 1, 2, 2])
_COLS = np.array([1, 3, 0, 1, 3])
_VALS = np.array([2., 4., 1., 3., 2.])
_SHAPE = (3, 4)


def _dense():
    m = np.zeros(_SHAPE, dtype=np.float32)
    for v, r, c in zip(_VALS, _ROWS, _COLS):
        m[r, c] = v
    return m


def _csr():
    indptr, indices, order = brainevent.coo2csr(_ROWS, _COLS, shape=_SHAPE)
    data = jnp.asarray(_VALS)[np.asarray(order)]
    return data, np.asarray(indices), np.asarray(indptr)


def test_csrmm_no_transpose_matches_dense():
    data, indices, indptr = _csr()
    B = jnp.arange(4 * 2, dtype=jnp.float32).reshape(4, 2)
    out = np.asarray(csrmm(data, indices, indptr, B, shape=_SHAPE, transpose=False))
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out, _dense() @ np.asarray(B), rtol=1e-5, atol=1e-5)


def test_csrmm_transpose_matches_dense():
    # transpose=True must compute Aᵀ @ B, where Aᵀ is (4, 3) and B is (3, 2).
    data, indices, indptr = _csr()
    B = jnp.arange(3 * 2, dtype=jnp.float32).reshape(3, 2)
    out = np.asarray(csrmm(data, indices, indptr, B, shape=_SHAPE, transpose=True))
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out, _dense().T @ np.asarray(B), rtol=1e-5, atol=1e-5)


def test_csrmm_transpose_grad_matches_dense():
    data, indices, indptr = _csr()
    B = jnp.arange(3 * 2, dtype=jnp.float32).reshape(3, 2)

    def f(d):
        return csrmm(d, indices, indptr, B, shape=_SHAPE, transpose=True).sum()

    g = np.asarray(jax.grad(f)(_csr()[0]))
    # dense reference gradient wrt the stored values
    dense_ref = _dense()

    def fd(flat):
        m = jnp.zeros(_SHAPE, dtype=jnp.float32)
        m = m.at[_ROWS, _COLS].set(flat)
        return (m.T @ B).sum()

    # values in CSR order correspond to coo2csr ``order``
    _, _, order = brainevent.coo2csr(_ROWS, _COLS, shape=_SHAPE)
    g_ref = np.asarray(jax.grad(fd)(jnp.asarray(_VALS)))[np.asarray(order)]
    np.testing.assert_allclose(g, g_ref, rtol=1e-5, atol=1e-5)
