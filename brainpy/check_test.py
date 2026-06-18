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

import jax

from brainpy import check as checking


class TestBoundChecks(unittest.TestCase):
    """Regression tests for P14-H2: eager bound checks must actually raise.

    ``is_float``/``is_integer`` route their ``min_bound``/``max_bound`` checks
    through ``jit_error_checking_no_args``. The previous implementation used a
    ``jax.pure_callback`` whose raise never propagated for a *concrete* (eager)
    predicate, so out-of-bound values were silently accepted.
    """

    def test_is_float_min_bound_raises(self):
        with self.assertRaises(Exception):
            checking.is_float(0.5, 'v', min_bound=1.0)

    def test_is_float_max_bound_raises(self):
        with self.assertRaises(Exception):
            checking.is_float(20.0, 'v', max_bound=10.0)

    def test_is_float_within_bounds_ok(self):
        self.assertEqual(checking.is_float(5.0, 'v', min_bound=1.0, max_bound=10.0), 5.0)

    def test_is_integer_min_bound_raises(self):
        with self.assertRaises(Exception):
            checking.is_integer(0, 'v', min_bound=1)

    def test_is_integer_max_bound_raises(self):
        with self.assertRaises(Exception):
            checking.is_integer(20, 'v', max_bound=10)

    def test_is_integer_within_bounds_ok(self):
        self.assertEqual(checking.is_integer(5, 'v', min_bound=1, max_bound=10), 5)

    def test_no_args_concrete_true_raises(self):
        with self.assertRaises(ValueError):
            checking.jit_error_checking_no_args(True, ValueError('boom'))

    def test_no_args_concrete_false_ok(self):
        # must not raise
        checking.jit_error_checking_no_args(False, ValueError('boom'))

    def test_no_args_under_jit_does_not_raise_at_trace(self):
        # When the predicate is a tracer (inside jit) the check must NOT raise
        # at trace time; it stays a deferred in-jit error signal.
        @jax.jit
        def f(x):
            checking.jit_error_checking_no_args(x > 1.0, ValueError('boom'))
            return x

        # tracing/compiling with a value that does not trip the predicate runs fine
        self.assertEqual(float(f(0.0)), 0.0)


class TestUtils(unittest.TestCase):
    def test_check_shape(self):
        all_shapes = [
            (1, 2, 3),
            (1, 4),
            (10, 2, 4)
        ]
        free_shape, fixed_shapes = checking.check_shape(all_shapes, free_axes=-1)
        self.assertEqual(free_shape, [3, 4, 4])
        self.assertEqual(fixed_shapes, [10, 2])

    def test_check_shape2(self):
        all_shapes = [
            (1, 2, 3, 8,),
            (10, 1, 4, 10),
            (10, 2, 4, 100)
        ]
        free_shape, fixed_shapes = checking.check_shape(all_shapes, free_axes=[2, -1])
        print(free_shape)
        print(fixed_shapes)
        self.assertEqual(free_shape, [[3, 8], [4, 10], [4, 100]])
        self.assertEqual(fixed_shapes, [10, 2])

    def test_check_shape3(self):
        all_shapes = [
            (1, 2, 3, 8,),
            (10, 1, 4, 10),
            (10, 2, 4, 100)
        ]
        free_shape, fixed_shapes = checking.check_shape(all_shapes, free_axes=[0, 2, -1])
        print(free_shape)
        print(fixed_shapes)
        self.assertEqual(free_shape, [[1, 3, 8], [10, 4, 10], [10, 4, 100]])
        self.assertEqual(fixed_shapes, [2])

    def test_check_shape4(self):
        all_shapes = [
            (1, 2, 3, 8,),
            (10, 1, 4, 10),
            (10, 2, 4, 100)
        ]
        with self.assertRaises(ValueError):
            free_shape, fixed_shapes = checking.check_shape(all_shapes, free_axes=[0, -1])
