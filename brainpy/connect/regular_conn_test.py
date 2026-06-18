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
import unittest

import numpy as np

import brainpy as bp
from brainpy import connect


class TestOne2One(unittest.TestCase):
    def test_one2one(self):
        for size in [100, (3, 4), (4, 5, 6)]:
            conn = connect.One2One()(pre_size=size, post_size=size)

            conn_mat, pre_ids, post_ids, pre2post, pre2syn, post2pre, post2syn = \
                conn.require('conn_mat', 'pre_ids', 'post_ids', 'pre2post', 'pre2syn', 'post2pre', 'post2syn')

            num = bp.tools.size2num(size)

            actual_mat = bp.math.zeros((num, num), dtype=bp.math.bool_)
            bp.math.fill_diagonal(actual_mat, True)

            assert bp.math.array_equal(actual_mat, conn_mat)
            assert bp.math.array_equal(pre_ids, bp.math.arange(num))
            assert bp.math.array_equal(post_ids, bp.math.arange(num))

            print('conn_mat', conn_mat)
            print('pre_ids', pre_ids)
            print('post_ids', post_ids)
            print('pre2post', pre2post)
            print('post2pre', post2pre)
            print('pre2syn', pre2syn)
            print('post2syn', post2syn)


class TestAll2All(unittest.TestCase):
    def test_all2all(self):
        for has_self in [True, False]:
            for size in [100, (3, 4), (4, 5, 6)]:
                conn = connect.All2All(include_self=has_self)(pre_size=size, post_size=size)
                mat = conn.require(connect.CONN_MAT)
                conn_mat, pre_ids, post_ids, pre2post, pre2syn, post2pre, post2syn = \
                    conn.require('conn_mat', 'pre_ids', 'post_ids', 'pre2post', 'pre2syn', 'post2pre', 'post2syn')
                num = bp.tools.size2num(size)

                print(mat)
                actual_mat = bp.math.ones((num, num), dtype=bp.math.bool_)
                if not has_self:
                    bp.math.fill_diagonal(actual_mat, False)
                assert bp.math.array_equal(actual_mat, mat)

                print()
                print('conn_mat', conn_mat)
                print('pre_ids', pre_ids)
                print('post_ids', post_ids)
                print('pre2post', pre2post)
                print('post2pre', post2pre)
                print('pre2syn', pre2syn)
                print('post2syn', post2syn)


class TestGridConn(unittest.TestCase):
    def test_grid_four(self):
        for periodic_boundary in [True, False]:
            for include_self in [True, False]:
                for size in (10, [10, 10], (4, 4, 5)):
                    conn = bp.conn.GridFour(include_self=include_self,
                                            periodic_boundary=periodic_boundary)(size, size)
                    mat = conn.build_mat()
                    pre_ids, post_ids = conn.build_coo()
                    new_mat = bp.math.zeros((np.prod(size), np.prod(size)), dtype=bool)
                    new_mat[pre_ids, post_ids] = True

                    print(f'periodic_boundary = {periodic_boundary}, include_self = {include_self}, size = {size}')
                    self.assertTrue(bp.math.allclose(mat, new_mat))

    def test_grid_eight(self):
        for periodic_boundary in [True, False]:
            for include_self in [True, False]:
                for size in (10, [10, 10], (4, 4, 5)):
                    conn = bp.conn.GridEight(include_self=include_self,
                                             periodic_boundary=periodic_boundary)(size, size)
                    mat = conn.build_mat()
                    pre_ids, post_ids = conn.build_coo()
                    new_mat = bp.math.zeros((np.prod(size), np.prod(size)), dtype=bool)
                    new_mat[pre_ids, post_ids] = True

                    print(f'periodic_boundary = {periodic_boundary}, include_self = {include_self}, size = {size}')
                    self.assertTrue(bp.math.allclose(mat, new_mat))

    def test_grid_N(self):
        for periodic_boundary in [True, False]:
            for include_self in [True, False]:
                for size in (10, [10, 10], (4, 4, 5)):
                    conn = bp.conn.GridN(include_self=include_self,
                                         periodic_boundary=periodic_boundary,
                                         N=2)(size, size)
                    mat = conn.build_mat()
                    pre_ids, post_ids = conn.build_coo()
                    new_mat = bp.math.zeros((np.prod(size), np.prod(size)), dtype=bool)
                    new_mat[pre_ids, post_ids] = True

                    print(f'periodic_boundary = {periodic_boundary}, include_self = {include_self}, size = {size}')
                    self.assertTrue(bp.math.allclose(mat, new_mat))
