# -*- coding: utf-8 -*-
"""Tests for ``brainpy/math/sparse/utils.py``.

Covers ``coo_to_csr``, ``csr_to_coo`` and ``csr_to_dense``. CSR results are
checked against dense numpy references built from the same COO triples, with
tiny matrices.
"""

import numpy as np
import pytest

import brainevent

from brainpy.math.sparse.utils import coo_to_csr, csr_to_coo, csr_to_dense


# ---------------------------------------------------------------------------
# coo_to_csr
# ---------------------------------------------------------------------------

def test_coo_to_csr_valid_roundtrip():
    # 3 x 4 matrix:
    #   [[0, 2, 0, 4],
    #    [1, 0, 0, 0],
    #    [0, 3, 0, 2]]
    pre = np.array([0, 0, 1, 2, 2])
    post = np.array([1, 3, 0, 1, 3])
    indices, indptr = coo_to_csr(pre, post, num_row=3)
    indices = np.asarray(indices)
    indptr = np.asarray(indptr)
    # CSR must be internally consistent.
    assert int(indptr[0]) == 0
    assert int(indptr[-1]) == len(indices) == pre.shape[0]
    assert np.asarray(indptr).dtype == np.int32
    np.testing.assert_array_equal(indptr, [0, 2, 3, 5])
    np.testing.assert_array_equal(indices, [1, 3, 0, 1, 3])


def test_coo_to_csr_unsorted():
    # Same matrix, but COO triples given in an unsorted (by row) order.
    pre = np.array([2, 0, 1, 0, 2])
    post = np.array([3, 1, 0, 3, 1])
    indices, indptr = coo_to_csr(pre, post, num_row=3)
    indptr = np.asarray(indptr)
    # Row pointers are independent of input order.
    np.testing.assert_array_equal(indptr, [0, 2, 3, 5])
    assert int(indptr[-1]) == np.asarray(indices).shape[0]


def test_coo_to_csr_empty_leading_and_middle_rows():
    # row0 empty, row1 -> col2, row2 empty, row3 -> cols {0, 1}
    pre = np.array([1, 3, 3])
    post = np.array([2, 0, 1])
    indices, indptr = coo_to_csr(pre, post, num_row=4)
    np.testing.assert_array_equal(np.asarray(indptr), [0, 0, 1, 1, 3])
    assert int(np.asarray(indptr)[-1]) == 3


def test_coo_to_csr_out_of_range_pre_id_raises():
    # ``3`` is out of range for ``num_row=3`` (valid rows are 0, 1, 2). Previously
    # the out-of-bounds scatter was silently dropped, yielding a corrupt CSR with
    # ``indptr[-1] != nse``. It must now raise instead of returning wrong output.
    pre = np.array([0, 1, 3])
    post = np.array([1, 2, 0])
    with pytest.raises(ValueError):
        coo_to_csr(pre, post, num_row=3)


def test_coo_to_csr_negative_pre_id_raises():
    pre = np.array([0, -1, 2])
    post = np.array([1, 2, 0])
    with pytest.raises(ValueError):
        coo_to_csr(pre, post, num_row=3)


# ---------------------------------------------------------------------------
# csr_to_coo
# ---------------------------------------------------------------------------

def _build_csr(rows, cols, shape):
    indptr, indices, order = brainevent.coo2csr(np.asarray(rows), np.asarray(cols), shape=shape)
    return np.asarray(indptr), np.asarray(indices), np.asarray(order)


def test_csr_to_coo_roundtrip():
    rows = np.array([0, 0, 1, 2, 2])
    cols = np.array([1, 3, 0, 1, 3])
    indptr, indices, _ = _build_csr(rows, cols, (3, 4))
    r, c = csr_to_coo(indices, indptr)
    np.testing.assert_array_equal(np.asarray(r), rows)
    np.testing.assert_array_equal(np.asarray(c), cols)


def test_csr_to_coo_empty_rows():
    # row0 empty, row1 -> col2, row2 empty, row3 -> cols {0, 1}
    indptr = np.array([0, 0, 1, 1, 3])
    indices = np.array([2, 0, 1])
    r, c = csr_to_coo(indices, indptr)
    np.testing.assert_array_equal(np.asarray(r), [1, 3, 3])
    np.testing.assert_array_equal(np.asarray(c), [2, 0, 1])


# ---------------------------------------------------------------------------
# csr_to_dense
# ---------------------------------------------------------------------------

def test_csr_to_dense_matches_reference():
    rows = np.array([0, 0, 1, 2, 2])
    cols = np.array([1, 3, 0, 1, 3])
    vals = np.array([2., 4., 1., 3., 2.])
    shape = (3, 4)
    ref = np.zeros(shape)
    for v, r, c in zip(vals, rows, cols):
        ref[r, c] = v
    indptr, indices, order = _build_csr(rows, cols, shape)
    data = vals[order]
    dense = np.asarray(csr_to_dense(data, indices, indptr, shape=shape))
    np.testing.assert_allclose(dense, ref, rtol=1e-6, atol=1e-6)
