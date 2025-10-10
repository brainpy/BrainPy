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
from functools import partial

import jax
from absl.testing import parameterized
from jax import vmap

import brainpy as bp
import brainpy.math as bm


class TestLoop(parameterized.TestCase):
    @parameterized.named_parameters(
        {"testcase_name": "_jit_scan={}_jit_f={}_unroll={}".format(jit_scan, jit_f, unroll),
         "jit_scan": jit_scan,
         "jit_f": jit_f,
         "unroll": unroll}
        for jit_scan in [False, True]
        for jit_f in [False, True]
        for unroll in [1, 2]
    )
    def test_for_loop(self, jit_scan, jit_f, unroll):
        rng = bm.random.RandomState(123)

        c = bm.Variable(rng.randn(4))
        d = rng.randn(2)
        all_a = rng.randn(5, 3)

        def f(a):
            assert a.shape == (3,)
            assert c.shape == (4,)
            b = bm.cos(bm.sum(bm.sin(a)) + bm.sum(bm.cos(c)) + bm.sum(bm.tan(d)))
            c.value = bm.sin(c * b)
            assert b.shape == ()
            return b

        if jit_f:
            f = bm.jit(f)
        scan = partial(bm.for_loop, f, unroll=unroll, )
        if jit_scan:
            scan = bm.jit(scan)
        ans = scan(operands=all_a)
        print(ans)
        print(c)

    def test_for_loop_progress_bar(self):
        xs = bm.arange(100)
        ys = bm.for_loop(lambda a: a, xs, progress_bar=True)
        self.assertTrue(bm.allclose(xs, ys))

    def test_for_loop2(self):
        class MyClass(bp.DynamicalSystem):
            def __init__(self):
                super().__init__()
                self.a = bm.Variable(bm.zeros(1))

            def update(self):
                self.a += 1

        cls = MyClass()
        indices = bm.arange(10)
        bm.for_loop(cls.step_run, indices)
        self.assertTrue(bm.allclose(cls.a, 10.))


class TestScan(unittest.TestCase):
    def test1(self):
        a = bm.Variable(1)

        def f(carray, x):
            carray += x
            a.value += 1.
            return carray, a.value

        carry, outs = bm.scan(f, bm.zeros(2), bm.arange(10))
        self.assertTrue(bm.allclose(carry, 45.))
        expected = bm.arange(1, 11).astype(outs.dtype)
        expected = bm.expand_dims(expected, axis=-1)
        self.assertTrue(bm.allclose(outs, expected))

    def test2(self):
        a = bm.Variable(1)

        def f(carray, x):
            carray += x
            a.value += 1.
            return carray, a.value

        @bm.jit
        def f_outer(carray, x):
            carry, outs = bm.scan(f, carray, x, unroll=2)
            return carry, outs

        carry, outs = f_outer(bm.zeros(2), bm.arange(10))
        self.assertTrue(bm.allclose(carry, 45.))
        expected = bm.arange(1, 11).astype(outs.dtype)
        expected = bm.expand_dims(expected, axis=-1)
        self.assertTrue(bm.allclose(outs, expected))

    def test_disable_jit(self):
        def cumsum(res, el):
            res = res + el
            print(res)
            return res, res  # ("carryover", "accumulated")

        a = bm.array([1, 2, 3, 5, 7, 11, 13, 17]).value
        result_init = 0
        with jax.disable_jit():
            final, result = jax.lax.scan(cumsum, result_init, a)

        b = bm.array([1, 2, 3, 5, 7, 11, 13, 17])
        result_init = 0
        with jax.disable_jit():
            final, result = bm.scan(cumsum, result_init, b)

    def test_array_aware_of_bp_array(self):
        def cumsum(res, el):
            res = bm.asarray(res + el)
            return res, res  # ("carryover", "accumulated")

        b = bm.array([1, 2, 3, 5, 7, 11, 13, 17])
        result_init = 0
        with jax.disable_jit():
            final, result = bm.scan(cumsum, result_init, b)


class TestCond(unittest.TestCase):
    def test1(self):
        bm.random.seed(1)
        bm.cond(True, lambda: bm.random.random(10), lambda: bm.random.random(10), ())
        bm.cond(False, lambda: bm.random.random(10), lambda: bm.random.random(10), ())


class TestIfElse(unittest.TestCase):
    def test1(self):
        def f(a):
            return bm.ifelse(conditions=[a < 0, a < 2, a < 5, a < 10, a < 20],
                             branches=[lambda: 1,
                                       lambda: 2,
                                       lambda: 3,
                                       lambda: 4,
                                       lambda: 5])

        self.assertTrue(f(3) == 3)
        self.assertTrue(f(1) == 2)
        self.assertTrue(f(-1) == 1)

    def test2(self):
        def f(a):
            return bm.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0, a > -1],
                             branches=[1, 2, 3, 4, 5])

        self.assertTrue(f(3) == 3)
        self.assertTrue(f(1) == 4)
        self.assertTrue(f(-1) == 5)

    def test_dyn_vars1(self):
        var_a = bm.Variable(bm.zeros(1))

        def f(a):
            def f1():
                var_a.value += 1
                return 1

            return bm.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0, a > -1],
                             branches=[f1,
                                       lambda: 2, lambda: 3,
                                       lambda: 4, lambda: 5])

        self.assertTrue(f(11) == 1)
        print(var_a.value)
        self.assertTrue(bm.all(var_a.value == 1))
        self.assertTrue(f(1) == 4)
        self.assertTrue(f(-1) == 5)

    def test_vmap(self):
        def f(operands):
            f = lambda a: bm.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
                                    branches=[lambda _: 1,
                                              lambda _: 2,
                                              lambda _: 3,
                                              lambda _: 4,
                                              lambda _: 5, ],
                                    operands=a)
            return vmap(f)(operands)

        r = f(bm.random.randint(-20, 20, 200))
        self.assertTrue(r.size == 200)

    def test_vmap2(self):
        def f2():
            f = lambda a: bm.ifelse(conditions=[a > 10, a > 5, a > 2, a > 0],
                                    branches=[1, 2, 3, 4, lambda _: 5],
                                    operands=a)
            return vmap(f)(bm.random.randint(-20, 20, 200))

        self.assertTrue(f2().size == 200)

    def test_grad1(self):
        def F2(x):
            return bm.ifelse(conditions=(x >= 10,),
                             branches=[lambda x: x,
                                       lambda x: x ** 2, ],
                             operands=x)

        self.assertTrue(bm.grad(F2)(9.0) == 18.)
        self.assertTrue(bm.grad(F2)(11.0) == 1.)

    def test_grad2(self):
        def F3(x):
            return bm.ifelse(conditions=(x >= 10, x >= 0),
                             branches=[lambda x: x,
                                       lambda x: x ** 2,
                                       lambda x: x ** 4, ],
                             operands=x)

        self.assertTrue(bm.grad(F3)(9.0) == 18.)
        self.assertTrue(bm.grad(F3)(11.0) == 1.)


class TestWhile(unittest.TestCase):
    def test1(self):
        bm.random.seed()

        a = bm.Variable(bm.zeros(1))
        b = bm.Variable(bm.ones(1))

        def cond(x, y):
            return x < 6.

        def body(x, y):
            a.value += x
            b.value *= y
            return x + b[0], y + 1.

        res = bm.while_loop(body, cond, operands=(1., 1.))
        print()
        print(res)

    def test2(self):
        bm.random.seed()

        a = bm.Variable(bm.zeros(1))
        b = bm.Variable(bm.ones(1))

        def cond(x, y):
            return x < 6.

        def body(x, y):
            a.value += x
            b.value *= y
            return x + b[0], y + 1.

        res = bm.while_loop(body, cond, operands=(1., 1.))
        print()
        print(res)

        with jax.disable_jit():
            a = bm.Variable(bm.zeros(1))
            b = bm.Variable(bm.ones(1))

            res2 = bm.while_loop(body, cond, operands=(1., 1.))
            print(res2)
            self.assertTrue(bm.array_equal(res2[0], res[0]))
            self.assertTrue(bm.array_equal(res2[1], res[1]))

    def test3(self):
        bm.random.seed()

        a = bm.Variable(bm.zeros(1))
        b = bm.Variable(bm.ones(1))

        def cond(x, y):
            return bm.all(a.value < 6.)

        def body(x, y):
            a.value += x
            b.value *= y

        res = bm.while_loop(body, cond, operands=(1., 1.))
        self.assertTrue(bm.allclose(a, 6.))
        self.assertTrue(bm.allclose(b, 1.))
        print()
        print(res)
        print(a)
        print(b)

    # def test4(self):
    #   bm.random.seed()
    #
    #   a = bm.Variable(bm.zeros(1))
    #   b = bm.Variable(bm.ones(1))
    #
    #   def cond(x, y):
    #     a.value += 1
    #     return bm.all(a.value < 6.)
    #
    #   def body(x, y):
    #     a.value += x
    #     b.value *= y
    #
    #   res = bm.while_loop(body, cond, operands=(1., 1.))
    #   self.assertTrue(bm.allclose(a, 7.))  # Corrected: condition function increments a each time before checking
    #   self.assertTrue(bm.allclose(b, 1.))
    #   print(res)
    #   print(a)
    #   print(b)
    #   print()

    def test5(self):
        bm.random.seed()

        a = bm.Variable(bm.zeros(1))
        b = bm.Variable(bm.ones(1))
        c = bm.Variable(bm.ones(1))

        def cond(x, y):
            a.value += 1
            return bm.all(a.value < 6.)

        def body(x, y):
            a.value += x
            b.value *= y
            return x + 1, y + 1

        @bm.jit
        def run(a, b):
            x, y = bm.while_loop(body, cond, operands=(a, b))
            return c + x

        # Test that JIT compilation fails when condition function has write states
        with self.assertRaises(ValueError) as cm:
            run(0., 1.)

        self.assertIn("cond_fun should not have any write states", str(cm.exception))
