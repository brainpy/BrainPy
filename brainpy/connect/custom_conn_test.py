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


class TestSparseMatConn(TestCase):
    def test_sparseMatConn(self):
        conn_mat = np.random.randint(2, size=(5, 3), dtype=bp.math.bool_)
        sparse_mat = csr_matrix(conn_mat)
        conn = bp.conn.SparseMatConn(sparse_mat)(pre_size=sparse_mat.shape[0], post_size=sparse_mat.shape[1])

        print(conn.requires('pre2post'))

        print(conn.requires('conn_mat'))
        print(csr_matrix.todense(sparse_mat))

        assert bp.math.array_equal(conn_mat, bp.math.asarray(csr_matrix.todense(sparse_mat), dtype=bp.math.bool_))
