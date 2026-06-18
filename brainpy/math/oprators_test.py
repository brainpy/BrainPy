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

import jax.numpy as jnp

import brainpy.math as bm


# class TestRegisterOP(unittest.TestCase):
#   def test_register_op(self):
#     from jax import jit
#     import brainpy as bp
#     bp.math.set_platform('cpu')
#
#     def abs_eval(*ins):
#       return ins
#
#     def custom_op(outs, ins):
#       y, y1 = outs
#       x, x2 = ins
#       y[:] = x + 1
#       y1[:] = x2 + 2
#
#     z = jnp.ones((1, 2), dtype=jnp.float32)
#     op = bm.register_op(op_name='add', cpu_func=custom_op, out_shapes=abs_eval, apply_cpu_func_to_gpu=True)
#     jit_op = jit(op)
#     print(jit_op(z, z))


class TestSyn2Post(unittest.TestCase):
    def test_syn2post_sum(self):
        data = bm.arange(5)
        segment_ids = bm.array([0, 0, 1, 1, 2])
        self.assertTrue(bm.array_equal(bm.syn2post_sum(data, segment_ids, 3),
                                       bm.asarray([1, 5, 4])))

    def test_syn2post_max(self):
        data = bm.arange(5)
        segment_ids = bm.array([0, 0, 1, 1, 2])
        self.assertTrue(bm.array_equal(bm.syn2post_max(data, segment_ids, 3),
                                       bm.asarray([1, 3, 4])))

    def test_syn2post_min(self):
        data = bm.arange(5)
        segment_ids = bm.array([0, 0, 1, 1, 2])
        self.assertTrue(bm.array_equal(bm.syn2post_min(data, segment_ids, 3),
                                       bm.asarray([0, 2, 4])))

    def test_syn2post_prod(self):
        data = bm.arange(5)
        segment_ids = bm.array([0, 0, 1, 1, 2])
        self.assertTrue(bm.array_equal(bm.syn2post_prod(data, segment_ids, 3),
                                       bm.asarray([0, 6, 4])))

    def test_syn2post_mean(self):
        data = bm.arange(5)
        segment_ids = bm.array([0, 0, 1, 1, 2])
        self.assertTrue(bm.array_equal(bm.syn2post_mean(data, segment_ids, 3),
                                       bm.asarray([0.5, 2.5, 4.])))

    def test_syn2post_softmax(self):
        data = bm.arange(5)
        segment_ids = bm.array([0, 0, 1, 1, 2])
        f_ans = bm.syn2post_softmax(data, segment_ids, 3)
        true_ans = bm.asarray([jnp.exp(data[0]) / (jnp.exp(data[0]) + jnp.exp(data[1])),
                               jnp.exp(data[1]) / (jnp.exp(data[0]) + jnp.exp(data[1])),
                               jnp.exp(data[2]) / (jnp.exp(data[2]) + jnp.exp(data[3])),
                               jnp.exp(data[3]) / (jnp.exp(data[2]) + jnp.exp(data[3])),
                               jnp.exp(data[4]) / jnp.exp(data[4])])
        print()
        print(bm.asarray(f_ans))
        print(true_ans)
        print(f_ans == true_ans)
        # self.assertTrue(bm.array_equal(bm.syn2post_softmax(data, segment_ids, 3),
        #                                true_ans))

        data = bm.arange(5)
        segment_ids = bm.array([0, 0, 1, 1, 2])
        print(bm.syn2post_softmax(data, segment_ids, 4))

#
# class TestSparseMatmul(unittest.TestCase):
#   def test_left_sparse_matmul1(self):
#     A = jnp.asarray([[0, 2, 0, 4],
#                      [1, 0, 0, 0],
#                      [0, 3, 0, 2]])
#     values = jnp.asarray([2, 4, 1, 3, 2])
#     rows = jnp.asarray([0, 0, 1, 2, 2])
#     columns = jnp.asarray([1, 3, 0, 1, 3])
#     sparse_A = {'data': values,  'index': (rows, columns), 'shape': (3, 4)}
#     # B = jnp.arange(8).reshape((4, 2))
#     B = jnp.arange(4)
#
#     self.assertTrue(bm.array_equal(bm.sparse_matmul(sparse_A, B),
#                                    jnp.dot(A, B)))
#
#   def test_left_sparse_matmul2(self):
#     A = jnp.asarray([[0, 2, 0, 4],
#                      [1, 0, 0, 0],
#                      [0, 3, 0, 2]])
#     values = jnp.asarray([2, 4, 1, 3, 2])
#     rows = jnp.asarray([0, 0, 1, 2, 2])
#     columns = jnp.asarray([1, 3, 0, 1, 3])
#     sparse_A = {'data': values, 'index': (rows, columns), 'shape': (3, 4)}
#     B = jnp.arange(8).reshape((4, 2))
#
#     self.assertTrue(bm.array_equal(bm.sparse_matmul(sparse_A, B),
#                                    jnp.dot(A, B)))
#
#   def test_right_sparse_matmul1(self):
#     B = jnp.asarray([[0, 2, 0, 4],
#                      [1, 0, 0, 0],
#                      [0, 3, 0, 2]])
#     values = jnp.asarray([2, 4, 1, 3, 2])
#     rows = jnp.asarray([0, 0, 1, 2, 2])
#     cols = jnp.asarray([1, 3, 0, 1, 3])
#     sparse_B = {'data': values, 'index': (rows, cols), 'shape': (3, 4)}
#
#     A = jnp.arange(6).reshape((2, 3))
#
#     self.assertTrue(bm.array_equal(bm.sparse_matmul(A, sparse_B),
#                                    jnp.dot(A, B)))
#
#   def test_right_sparse_matmul2(self):
#     B = jnp.asarray([[0, 2, 0, 4],
#                      [1, 0, 0, 0],
#                      [0, 3, 0, 2]])
#     values = jnp.asarray([2, 4, 1, 3, 2])
#     rows = jnp.asarray([0, 0, 1, 2, 2])
#     cols = jnp.asarray([1, 3, 0, 1, 3])
#     sparse_B = {'data': values, 'index': (rows, cols), 'shape': (3, 4)}
#     A = jnp.arange(3)
#
#     print(bm.sparse_matmul(A, sparse_B))
#     print(jnp.dot(A, B))
#
#     self.assertTrue(bm.array_equal(bm.sparse_matmul(A, sparse_B),
#                                    jnp.dot(A, B)))
