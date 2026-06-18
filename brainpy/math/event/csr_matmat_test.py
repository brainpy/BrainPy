# -*- coding: utf-8 -*-
"""Regression tests for ``brainpy/math/event/csr_matmat.py``.

Guards the event-driven (binary) CSR matmat, especially the ``transpose=True``
branch (must compute ``Aᵀ @ E``), against a dense numpy reference.
"""

import jax.numpy as jnp
import numpy as np

import brainevent

from brainpy.math.event.csr_matmat import csrmm


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


def test_event_csrmm_no_transpose_matches_dense():
    data, indices, indptr = _csr()
    E = np.array([[True, False], [False, True], [True, True], [False, False]])
    out = np.asarray(csrmm(data, indices, indptr, jnp.asarray(E), shape=_SHAPE, transpose=False))
    assert out.shape == (3, 2)
    np.testing.assert_allclose(out, _dense() @ E.astype(np.float32), rtol=1e-5, atol=1e-5)


def test_event_csrmm_transpose_matches_dense():
    # transpose=True must compute Aᵀ @ E (Aᵀ is (4,3), E is (3,2)).
    data, indices, indptr = _csr()
    E = np.array([[True, False], [False, True], [True, True]])
    out = np.asarray(csrmm(data, indices, indptr, jnp.asarray(E), shape=_SHAPE, transpose=True))
    assert out.shape == (4, 2)
    np.testing.assert_allclose(out, _dense().T @ E.astype(np.float32), rtol=1e-5, atol=1e-5)


def test_event_csrmm_matches_float_csrmm_with_binary_input():
    # An all-True event matrix multiplied by the binary path equals the dense
    # product restricted to the selected entries.
    data, indices, indptr = _csr()
    E = np.array([[True, True], [True, True], [True, True], [True, True]])
    out = np.asarray(csrmm(data, indices, indptr, jnp.asarray(E), shape=_SHAPE, transpose=False))
    np.testing.assert_allclose(out, _dense() @ E.astype(np.float32), rtol=1e-5, atol=1e-5)
