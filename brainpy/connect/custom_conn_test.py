# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from unittest import TestCase

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import brainpy as bp


class TestIJConn(TestCase):
    def test_ij(self):
        conn = bp.connect.IJConn(i=np.array([0, 1, 2]),
                                 j=np.array([0, 0, 0]))(pre_size=5, post_size=3)

        pre2post, post2pre, conn_mat = conn.requires('pre2post', 'post2pre', 'conn_mat')

        assert bp.math.array_equal(pre2post[0], bp.math.array([0, 0, 0]))
        assert bp.math.array_equal(post2pre[0], bp.math.array([0, 1, 2]))

        a = bp.math.array([[True, False, False],
                           [True, False, False],
                           [True, False, False],
                           [False, False, False],
                           [False, False, False]])
        print()
        print('conn_mat', conn_mat)
        assert bp.math.array_equal(conn_mat, a)


class TestMatConn(TestCase):
    def test_MatConn1(self):
        bp.math.random.seed(123)
        actual_mat = np.random.randint(2, size=(5, 3), dtype=bp.math.bool_)
        conn = bp.connect.MatConn(conn_mat=actual_mat)(pre_size=5, post_size=3)

        pre2post, post2pre, conn_mat = conn.requires('pre2post', 'post2pre', 'conn_mat')

        print()
        print('conn_mat', conn_mat)

        assert bp.math.array_equal(conn_mat, actual_mat)

    def test_MatConn2(self):
        conn = bp.connect.MatConn(conn_mat=np.random.randint(2, size=(5, 3), dtype=bp.math.bool_))
        with pytest.raises(AssertionError):
            conn(pre_size=5, post_size=1)


class TestCSRConn(TestCase):
    def _csr(self):
        # 3 pre-synaptic neurons (indptr has 4 entries), max post id == 2
        indices = np.array([0, 1, 2, 0, 1], dtype=np.int32)
        indptr = np.array([0, 2, 3, 5], dtype=np.int32)
        return indices, indptr

    def test_csrconn_consistent_ok(self):
        # P16-H2: a CSRConn whose declared pre_size matches the indptr length
        # must build without error.
        indices, indptr = self._csr()
        conn = bp.conn.CSRConn(indices, indptr)
        ind, ip = conn.require(3, 3, 'csr')
        assert np.array_equal(np.asarray(ind), indices)
        assert np.array_equal(np.asarray(ip), indptr)

    def test_csrconn_inconsistent_pre_num_raises(self):
        # P16-H2: previously the guard ``self.pre_num != self.pre_num`` was a
        # tautology (always False), so an inconsistent pre_size silently produced
        # a malformed CSR. It must now raise.
        indices, indptr = self._csr()  # indptr implies 3 pre
        conn = bp.conn.CSRConn(indices, indptr)
        with pytest.raises(bp.errors.ConnectorError):
            conn.require(5, 3, 'csr')  # pre=5 inconsistent with indptr (3)

    def test_coo2csr_no_dtype_warning(self):
        # P16-M1: coo2csr must not emit the int32->uint32 scatter FutureWarning
        # (which is slated to become an error in future JAX).
        import warnings
        import jax.numpy as jnp
        from brainpy.connect.base import coo2csr
        pre = jnp.array([0, 0, 1, 2, 2, 2])
        post = jnp.array([1, 2, 0, 0, 1, 2])
        with warnings.catch_warnings():
            warnings.simplefilter('error', FutureWarning)
            ind, indptr = coo2csr((pre, post), 3)
        assert np.array_equal(np.asarray(indptr), np.array([0, 2, 3, 6]))


class TestSparseMatConn(TestCase):
    def test_sparseMatConn(self):
        conn_mat = np.random.randint(2, size=(5, 3), dtype=bp.math.bool_)
        sparse_mat = csr_matrix(conn_mat)
        conn = bp.conn.SparseMatConn(sparse_mat)(pre_size=sparse_mat.shape[0], post_size=sparse_mat.shape[1])

        print(conn.requires('pre2post'))

        print(conn.requires('conn_mat'))
        print(csr_matrix.todense(sparse_mat))

        assert bp.math.array_equal(conn_mat, bp.math.asarray(csr_matrix.todense(sparse_mat), dtype=bp.math.bool_))
