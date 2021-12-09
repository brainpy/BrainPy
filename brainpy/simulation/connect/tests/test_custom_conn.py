# -*- coding: utf-8 -*-
import math
from unittest import TestCase

import numpy as np
import pytest
import brainpy as bp
from scipy.sparse import csr_matrix

MAT_DTYPE = np.bool_
IDX_DTYPE = np.uint32


class TestJIConn(TestCase):
    def test_ij(self):
        conn = bp.connect.IJConn(i=np.array([0, 1, 2]),
                                 j=np.array([0, 0, 0]))
        conn = conn(pre_size=5, post_size=3)

        pre2post = conn.requires('pre2post')
        assert bp.math.array_equal(pre2post[0], bp.math.array([0, 0, 0]))

        post2pre = conn.requires('post2pre')
        assert bp.math.array_equal(post2pre[0], bp.math.array([0, 1, 2]))

        a = bp.math.array([[True, False, False],
                           [True, False, False],
                           [True, False, False],
                           [False, False, False],
                           [False, False, False]])

        assert bp.math.array_equal(conn.require(bp.conn.CONN_MAT), a)


class TestMatConn(TestCase):
    def test_MatConn1(self):
        bp.math.random.seed(123)
        conn = bp.connect.MatConn(conn_mat=np.random.randint(2, size=(5, 3), dtype=bp.math.bool_))
        conn = conn(pre_size=5, post_size=3)

        print(conn.requires('pre2post'))
        print(conn.requires(bp.connect.CONN_MAT))

    def test_MatConn2(self):
        conn = bp.connect.MatConn(conn_mat=np.random.randint(2, size=(5, 3), dtype=bp.math.bool_))
        with pytest.raises(AssertionError):
            conn = conn(pre_size=5, post_size=1)

class TestSparseMatConn(TestCase):
    def test_sparseMatConn(self):
        conn_mat = np.random.randint(2, size=(5, 3), dtype=bp.math.bool_)
        sparse_mat = csr_matrix(conn_mat)
        conn = bp.conn.SparseMatConn(sparse_mat)
        conn = conn(pre_size=sparse_mat.shape[0], post_size=sparse_mat.shape[1])

        pre2syn = conn.require('pre2syn')
        assert bp.math.array_equal(pre2syn[0], conn.pre2syn[0])
        assert bp.math.array_equal(pre2syn[1], conn.pre2syn[1])

        print(conn.requires('pre2post'))
        print(conn.requires(bp.connect.CONN_MAT))
