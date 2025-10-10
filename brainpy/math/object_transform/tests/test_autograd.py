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
from pprint import pprint

import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm


class TestPureFuncGrad(unittest.TestCase):
    def test_grad_pure_func_1(self):
        def call(a, b, c): return bm.sum(a + b + c)

        bm.random.seed(1)
        a = bm.ones(10)
        b = bm.random.randn(10)
        c = bm.random.uniform(size=10)
        f_grad = bm.grad(call, argnums=[0, 1, 2])
        grads = f_grad(a, b, c)

        for g in grads: assert (g == 1.).all()

    def test_grad_pure_func_2(self):
        def call(a, b, c): return bm.sum(a + b + c)

        bm.random.seed(1)
        a = bm.ones(10)
        b = bm.random.randn(10)
        c = bm.random.uniform(size=10)
        f_grad = bm.grad(call)
        assert (f_grad(a, b, c) == 1.).all()

    def test_grad_pure_func_aux1(self):
        def call(a, b, c):
            return bm.sum(a + b + c), (bm.sin(100), bm.exp(0.1))

        bm.random.seed(1)
        f_grad = bm.grad(call, argnums=[0, 1, 2])
        with pytest.raises(TypeError):
            f_grad(bm.ones(10), bm.random.randn(10), bm.random.uniform(size=10))

    def test_grad_pure_func_aux2(self):
        def call(a, b, c):
            return bm.sum(a + b + c), (bm.sin(100), bm.exp(0.1))

        bm.random.seed(1)
        f_grad = bm.grad(call, argnums=[0, 1, 2], has_aux=True)
        grads, aux = f_grad(bm.ones(10), bm.random.randn(10), bm.random.uniform(size=10))
        for g in grads: assert (g == 1.).all()
        assert aux[0] == bm.sin(100)
        assert aux[1] == bm.exp(0.1)

    def test_grad_pure_func_return1(self):
        def call(a, b, c): return bm.sum(a + b + c)

        bm.random.seed(1)
        a = bm.ones(10)
        b = bm.random.randn(10)
        c = bm.random.uniform(size=10)
        f_grad = bm.grad(call, return_value=True)
        grads, returns = f_grad(a, b, c)
        assert (grads == 1.).all()
        assert returns == bm.sum(a + b + c)

    def test_grad_func_return_aux1(self):
        def call(a, b, c):
            return bm.sum(a + b + c), (bm.sin(100), bm.exp(0.1))

        bm.random.seed(1)
        a = bm.ones(10)
        b = bm.random.randn(10)
        c = bm.random.uniform(size=10)
        f_grad = bm.grad(call, return_value=True, has_aux=True)
        grads, returns, aux = f_grad(a, b, c)
        assert (grads == 1.).all()
        assert returns == bm.sum(a + b + c)
        assert aux[0] == bm.sin(100)
        assert aux[1] == bm.exp(0.1)

    def test_grad_jit(self):
        def call(a, b, c): return bm.sum(a + b + c)

        bm.random.seed(1)
        a = bm.ones(10)
        b = bm.random.randn(10)
        c = bm.random.uniform(size=10)
        f_grad = bm.jit(bm.grad(call))
        assert (f_grad(a, b, c) == 1.).all()


class TestObjectFuncGrad(unittest.TestCase):
    def test_grad_ob1(self):
        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()

                self.a = bm.TrainVar(bm.ones(10))
                self.b = bm.TrainVar(bm.random.randn(10))
                self.c = bm.TrainVar(bm.random.uniform(size=10))

            def __call__(self):
                return bm.sum(self.a + self.b + self.c)

        bm.random.seed(0)

        t = Test()
        f_grad = bm.grad(t, grad_vars={'a': t.a, 'b': t.b, 'c': t.c})
        grads = f_grad()
        for g in grads.values():
            assert (g == 1.).all()

        t = Test()
        f_grad = bm.grad(t, grad_vars=[t.a, t.b])
        grads = f_grad()
        for g in grads: assert (g == 1.).all()

        t = Test()
        f_grad = bm.grad(t, grad_vars=t.a)
        grads = f_grad()
        assert (grads == 1.).all()

    def test_grad_ob_aux(self):
        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()
                self.a = bm.TrainVar(bm.ones(10))
                self.b = bm.TrainVar(bm.random.randn(10))
                self.c = bm.TrainVar(bm.random.uniform(size=10))

            def __call__(self):
                return bm.sum(self.a + self.b + self.c), (bm.sin(100), bm.exp(0.1))

        bm.random.seed(0)
        t = Test()
        f_grad = bm.grad(t, grad_vars=[t.a, t.b], has_aux=True)
        grads, aux = f_grad()
        for g in grads: assert (g == 1.).all()
        assert aux[0] == bm.sin(100)
        assert aux[1] == bm.exp(0.1)

        t = Test()
        f_grad = bm.grad(t, grad_vars=t.a, has_aux=True)
        grads, aux = f_grad()
        assert (grads == 1.).all()
        assert aux[0] == bm.sin(100)
        assert aux[1] == bm.exp(0.1)

    def test_grad_ob_return(self):
        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()
                self.a = bm.TrainVar(bm.ones(10))
                self.b = bm.TrainVar(bm.random.randn(10))
                self.c = bm.TrainVar(bm.random.uniform(size=10))

            def __call__(self):
                return bm.sum(self.a + self.b + self.c)

        bm.random.seed(0)
        t = Test()
        f_grad = bm.grad(t, grad_vars=[t.a, t.b], return_value=True)
        grads, returns = f_grad()
        for g in grads: assert (g == 1.).all()
        assert returns == t()

        t = Test()
        f_grad = bm.grad(t, grad_vars=t.a, return_value=True)
        grads, returns = f_grad()
        assert (grads == 1.).all()
        assert returns == t()

    def test_grad_ob_aux_return(self):
        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()
                self.a = bm.TrainVar(bm.ones(10))
                self.b = bm.TrainVar(bm.random.randn(10))
                self.c = bm.TrainVar(bm.random.uniform(size=10))

            def __call__(self):
                return bm.sum(self.a + self.b + self.c), (bm.sin(100), bm.exp(0.1))

        bm.random.seed(0)
        t = Test()
        f_grad = bm.grad(t, grad_vars=[t.a, t.b],
                         has_aux=True, return_value=True)
        grads, returns, aux = f_grad()
        for g in grads: assert (g == 1.).all()
        assert returns == bm.sum(t.a + t.b + t.c)
        assert aux[0] == bm.sin(100)
        assert aux[1] == bm.exp(0.1)

        t = Test()
        f_grad = bm.grad(t, grad_vars=t.a,
                         has_aux=True, return_value=True)
        grads, returns, aux = f_grad()
        assert (grads == 1.).all()
        assert returns == bm.sum(t.a + t.b + t.c)
        assert aux[0] == bm.sin(100)
        assert aux[1] == bm.exp(0.1)

    def test_grad_ob_argnums(self):
        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()
                bp.math.random.seed()
                self.a = bm.TrainVar(bm.ones(10))
                self.b = bm.TrainVar(bm.random.randn(10))
                self.c = bm.TrainVar(bm.random.uniform(size=10))

            def __call__(self, d):
                return bm.sum(self.a + self.b + self.c + 2 * d)

        bm.random.seed(0)

        t = Test()
        f_grad = bm.grad(t, t.vars(), argnums=0)
        var_grads, arg_grads = f_grad(bm.random.random(10))
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads == 2.).all()

        t = Test()
        f_grad = bm.grad(t, t.vars(), argnums=[0])
        var_grads, arg_grads = f_grad(bm.random.random(10))
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads[0] == 2.).all()

        t = Test()
        f_grad = bm.grad(t, argnums=0)
        arg_grads = f_grad(bm.random.random(10))
        assert (arg_grads == 2.).all()

        t = Test()
        f_grad = bm.grad(t, argnums=[0])
        arg_grads = f_grad(bm.random.random(10))
        assert (arg_grads[0] == 2.).all()

    def test_grad_ob_argnums_aux(self):
        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()
                self.a = bm.TrainVar(bm.ones(10))
                self.b = bm.TrainVar(bm.random.randn(10))
                self.c = bm.TrainVar(bm.random.uniform(size=10))

            def __call__(self, d):
                return bm.sum(self.a + self.b + self.c + 2 * d), (bm.sin(100), bm.exp(0.1))

        bm.random.seed(0)

        t = Test()
        f_grad = bm.grad(t, grad_vars=t.vars(), argnums=0, has_aux=True)
        (var_grads, arg_grads), aux = f_grad(bm.random.random(10))
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads == 2.).all()
        assert aux[0] == bm.sin(100)
        assert aux[1] == bm.exp(0.1)

        t = Test()
        f_grad = bm.grad(t, grad_vars=t.vars(), argnums=[0], has_aux=True)
        (var_grads, arg_grads), aux = f_grad(bm.random.random(10))
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads[0] == 2.).all()
        assert aux[0] == bm.sin(100)
        assert aux[1] == bm.exp(0.1)

        t = Test()
        f_grad = bm.grad(t, argnums=0, has_aux=True)
        arg_grads, aux = f_grad(bm.random.random(10))
        assert (arg_grads == 2.).all()
        assert aux[0] == bm.sin(100)
        assert aux[1] == bm.exp(0.1)

        t = Test()
        f_grad = bm.grad(t, argnums=[0], has_aux=True)
        arg_grads, aux = f_grad(bm.random.random(10))
        assert (arg_grads[0] == 2.).all()
        assert aux[0] == bm.sin(100)
        assert aux[1] == bm.exp(0.1)

    def test_grad_ob_argnums_return(self):
        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()

                self.a = bm.TrainVar(bm.ones(10))
                self.b = bm.TrainVar(bm.random.randn(10))
                self.c = bm.TrainVar(bm.random.uniform(size=10))

            def __call__(self, d):
                return bm.sum(self.a + self.b + self.c + 2 * d)

        bm.random.seed(0)

        t = Test()
        f_grad = bm.grad(t, t.vars(), argnums=0, return_value=True)
        d = bm.random.random(10)
        (var_grads, arg_grads), loss = f_grad(d)
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads == 2.).all()
        assert loss == t(d)

        t = Test()
        f_grad = bm.grad(t, t.vars(), argnums=[0], return_value=True)
        d = bm.random.random(10)
        (var_grads, arg_grads), loss = f_grad(d)
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads[0] == 2.).all()
        assert loss == t(d)

        t = Test()
        f_grad = bm.grad(t, argnums=0, return_value=True)
        d = bm.random.random(10)
        arg_grads, loss = f_grad(d)
        assert (arg_grads == 2.).all()
        assert loss == t(d)

        t = Test()
        f_grad = bm.grad(t, argnums=[0], return_value=True)
        d = bm.random.random(10)
        arg_grads, loss = f_grad(d)
        assert (arg_grads[0] == 2.).all()
        assert loss == t(d)

    def test_grad_ob_argnums_aux_return(self):
        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()
                self.a = bm.TrainVar(bm.ones(10))
                self.b = bm.TrainVar(bm.random.randn(10))
                self.c = bm.TrainVar(bm.random.uniform(size=10))

            def __call__(self, d):
                return bm.sum(self.a + self.b + self.c + 2 * d), (bm.sin(100), bm.exp(0.1))

        bm.random.seed(0)

        t = Test()
        f_grad = bm.grad(t, grad_vars=t.vars(), argnums=0, has_aux=True, return_value=True)
        d = bm.random.random(10)
        (var_grads, arg_grads), loss, aux = f_grad(d)
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads == 2.).all()
        assert aux[0] == bm.sin(100)
        assert aux[1] == bm.exp(0.1)
        assert loss == t(d)[0]

        t = Test()
        f_grad = bm.grad(t, grad_vars=t.vars(), argnums=[0], has_aux=True, return_value=True)
        d = bm.random.random(10)
        (var_grads, arg_grads), loss, aux = f_grad(d)
        for g in var_grads.values(): assert (g == 1.).all()
        assert (arg_grads[0] == 2.).all()
        assert aux[0] == bm.sin(100)
        assert aux[1] == bm.exp(0.1)
        assert loss == t(d)[0]

        t = Test()
        f_grad = bm.grad(t, argnums=0, has_aux=True, return_value=True)
        d = bm.random.random(10)
        arg_grads, loss, aux = f_grad(d)
        assert (arg_grads == 2.).all()
        assert aux[0] == bm.sin(100)
        assert aux[1] == bm.exp(0.1)
        assert loss == t(d)[0]

        t = Test()
        f_grad = bm.grad(t, argnums=[0], has_aux=True, return_value=True)
        d = bm.random.random(10)
        arg_grads, loss, aux = f_grad(d)
        assert (arg_grads[0] == 2.).all()
        assert aux[0] == bm.sin(100)
        assert aux[1] == bm.exp(0.1)
        assert loss == t(d)[0]


# class TestPureFuncJacobian(unittest.TestCase):
#   def test1(self):
#     jac, aux = _jacfwd(lambda x: (x ** 3, [x ** 2]), has_aux=True)(3.)
#     self.assertTrue(jax.numpy.allclose(jac, jax.jacfwd(lambda x: x ** 3)(3.)))
#     self.assertTrue(aux[0] == 9.)
#
#   def test_jacfwd_and_aux_nested(self):
#     def f(x):
#       jac, aux = _jacfwd(lambda x: (x ** 3, [x ** 3]), has_aux=True)(x)
#       return aux[0]
#
#     f2 = lambda x: x ** 3
#
#     self.assertEqual(_jacfwd(f)(4.), _jacfwd(f2)(4.))
#     self.assertEqual(jax.jit(_jacfwd(f))(4.), _jacfwd(f2)(4.))
#     self.assertEqual(jax.jit(_jacfwd(jax.jit(f)))(4.), _jacfwd(f2)(4.))
#
#     self.assertEqual(_jacfwd(f)(bm.asarray(4.)), _jacfwd(f2)(bm.asarray(4.)))
#     self.assertEqual(jax.jit(_jacfwd(f))(bm.asarray(4.)), _jacfwd(f2)(bm.asarray(4.)))
#     self.assertEqual(jax.jit(_jacfwd(jax.jit(f)))(bm.asarray(4.)), _jacfwd(f2)(bm.asarray(4.)))
#
#     def f(x):
#       jac, aux = _jacfwd(lambda x: (x ** 3, [x ** 3]), has_aux=True)(x)
#       return aux[0] * bm.sin(x)
#
#     f2 = lambda x: x ** 3 * bm.sin(x)
#
#     self.assertEqual(_jacfwd(f)(4.), _jacfwd(f2)(4.))
#     self.assertEqual(jax.jit(_jacfwd(f))(4.), _jacfwd(f2)(4.))
#     self.assertEqual(jax.jit(_jacfwd(jax.jit(f)))(4.), _jacfwd(f2)(4.))
#
#     self.assertEqual(_jacfwd(f)(bm.asarray(4.)), _jacfwd(f2)(bm.asarray(4.)))
#     self.assertEqual(jax.jit(_jacfwd(f))(bm.asarray(4.)), _jacfwd(f2)(bm.asarray(4.)))
#     self.assertEqual(jax.jit(_jacfwd(jax.jit(f)))(bm.asarray(4.)), _jacfwd(f2)(bm.asarray(4.)))
#
#   def test_jacrev1(self):
#     def f1(x, y):
#       r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
#       return r
#
#     br = bm.jacrev(f1)(bm.array([1., 2., 3.]), bm.array([10., 5.]))
#     jr = jax.jacrev(f1)(bm.array([1., 2., 3.]), bm.array([10., 5.]))
#     assert (br == jr).all()
#
#     br = bm.jacrev(f1, argnums=(0, 1))(bm.array([1., 2., 3.]), bm.array([10., 5.]))
#     jr = jax.jacrev(f1, argnums=(0, 1))(bm.array([1., 2., 3.]), bm.array([10., 5.]))
#     assert (br[0] == jr[0]).all()
#     assert (br[1] == jr[1]).all()
#
#   def test_jacrev2(self):
#     print()
#
#     def f2(x, y):
#       r1 = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1]])
#       r2 = jnp.asarray([4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
#       return r1, r2
#
#     jr = jax.jacrev(f2)(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
#     pprint(jr)
#
#     br = bm.jacrev(f2)(bm.array([1., 2., 3.]).value, bm.array([10., 5.]).value)
#     pprint(br)
#     assert bm.array_equal(br[0], jr[0])
#     assert bm.array_equal(br[1], jr[1])
#
#     br = bm.jacrev(f2)(bm.array([1., 2., 3.]), bm.array([10., 5.]))
#     pprint(br)
#     assert bm.array_equal(br[0], jr[0])
#     assert bm.array_equal(br[1], jr[1])
#
#     def f2(x, y):
#       r1 = bm.asarray([x[0] * y[0], 5 * x[2] * y[1]])
#       r2 = bm.asarray([4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
#       return r1, r2
#
#     br = bm.jacrev(f2)(bm.array([1., 2., 3.]).value, bm.array([10., 5.]).value)
#     pprint(br)
#     assert bm.array_equal(br[0], jr[0])
#     assert bm.array_equal(br[1], jr[1])
#
#     br = bm.jacrev(f2)(bm.array([1., 2., 3.]), bm.array([10., 5.]))
#     pprint(br)
#     assert bm.array_equal(br[0], jr[0])
#     assert bm.array_equal(br[1], jr[1])
#
#   def test_jacrev3(self):
#     print()
#
#     def f3(x, y):
#       r1 = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1]])
#       r2 = jnp.asarray([4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
#       return r1, r2
#
#     jr = jax.jacrev(f3, argnums=(0, 1))(jnp.array([1., 2., 3.]), jnp.array([10., 5.]))
#     pprint(jr)
#
#     br = bm.jacrev(f3, argnums=(0, 1))(bm.array([1., 2., 3.]).value, bm.array([10., 5.]).value)
#     pprint(br)
#     assert bm.array_equal(br[0][0], jr[0][0])
#     assert bm.array_equal(br[0][1], jr[0][1])
#     assert bm.array_equal(br[1][0], jr[1][0])
#     assert bm.array_equal(br[1][1], jr[1][1])
#
#     br = bm.jacrev(f3, argnums=(0, 1))(bm.array([1., 2., 3.]), bm.array([10., 5.]))
#     pprint(br)
#     assert bm.array_equal(br[0][0], jr[0][0])
#     assert bm.array_equal(br[0][1], jr[0][1])
#     assert bm.array_equal(br[1][0], jr[1][0])
#     assert bm.array_equal(br[1][1], jr[1][1])
#
#     def f3(x, y):
#       r1 = bm.asarray([x[0] * y[0], 5 * x[2] * y[1]])
#       r2 = bm.asarray([4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
#       return r1, r2
#
#     br = bm.jacrev(f3, argnums=(0, 1))(bm.array([1., 2., 3.]).value, bm.array([10., 5.]).value)
#     pprint(br)
#     assert bm.array_equal(br[0][0], jr[0][0])
#     assert bm.array_equal(br[0][1], jr[0][1])
#     assert bm.array_equal(br[1][0], jr[1][0])
#     assert bm.array_equal(br[1][1], jr[1][1])
#
#     br = bm.jacrev(f3, argnums=(0, 1))(bm.array([1., 2., 3.]), bm.array([10., 5.]))
#     pprint(br)
#     assert bm.array_equal(br[0][0], jr[0][0])
#     assert bm.array_equal(br[0][1], jr[0][1])
#     assert bm.array_equal(br[1][0], jr[1][0])
#     assert bm.array_equal(br[1][1], jr[1][1])
#
#   def test_jacrev_aux1(self):
#     x = bm.array([1., 2., 3.])
#     y = bm.array([10., 5.])
#
#     def f1(x, y):
#       a = 4 * x[1] ** 2 - 2 * x[2]
#       r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], a, x[2] * jnp.sin(x[0])])
#       return r, a
#
#     f2 = lambda *args: f1(*args)[0]
#     jr = jax.jacrev(f2)(x, y)  # jax jacobian
#     pprint(jr)
#     grads, aux = bm.jacrev(f1, has_aux=True)(x, y)
#     assert (grads == jr).all()
#     assert aux == (4 * x[1] ** 2 - 2 * x[2])
#
#     jr = jax.jacrev(f2, argnums=(0, 1))(x, y)  # jax jacobian
#     pprint(jr)
#     grads, aux = bm.jacrev(f1, argnums=(0, 1), has_aux=True)(x, y)
#     assert (grads[0] == jr[0]).all()
#     assert (grads[1] == jr[1]).all()
#     assert aux == (4 * x[1] ** 2 - 2 * x[2])
#
#   def test_jacrev_return_aux1(self):
#     bm.enable_x64()
#
#     def f1(x, y):
#       a = 4 * x[1] ** 2 - 2 * x[2]
#       r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], a, x[2] * jnp.sin(x[0])])
#       return r, a
#
#     _x = bm.array([1., 2., 3.])
#     _y = bm.array([10., 5.])
#     _r, _a = f1(_x, _y)
#     f2 = lambda *args: f1(*args)[0]
#     _g1 = jax.jacrev(f2)(_x, _y)  # jax jacobian
#     pprint(_g1)
#     _g2 = jax.jacrev(f2, argnums=(0, 1))(_x, _y)  # jax jacobian
#     pprint(_g2)
#
#     grads, vec, aux = bm.jacrev(f1, return_value=True, has_aux=True)(_x, _y)
#     assert (grads == _g1).all()
#     assert aux == _a
#     assert (vec == _r).all()
#
#     grads, vec, aux = bm.jacrev(f1, return_value=True, argnums=(0, 1), has_aux=True)(_x, _y)
#     assert (grads[0] == _g2[0]).all()
#     assert (grads[1] == _g2[1]).all()
#     assert aux == _a
#     assert (vec == _r).all()
#
#     bm.disable_x64()


class TestClassFuncJacobian(unittest.TestCase):
    def test_jacrev1(self):
        def f1(x, y):
            r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
            return r

        _x = bm.array([1., 2., 3.])
        _y = bm.array([10., 5.])

        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()
                self.x = bm.Variable(bm.array([1., 2., 3.]))
                self.y = bm.Variable(bm.array([10., 5.]))

            def __call__(self, ):
                a = self.x[0] * self.y[0]
                b = 5 * self.x[2] * self.y[1]
                c = 4 * self.x[1] ** 2 - 2 * self.x[2]
                d = self.x[2] * jnp.sin(self.x[0])
                r = jnp.asarray([a, b, c, d])
                return r

        _jr = jax.jacrev(f1)(_x, _y)
        t = Test()
        br = bm.jacrev(t, grad_vars=t.x)()
        self.assertTrue((br == _jr).all())

        _jr = jax.jacrev(f1, argnums=(0, 1))(_x, _y)
        t = Test()
        br = bm.jacrev(t, grad_vars=[t.x, t.y])()
        self.assertTrue((br[0] == _jr[0]).all())
        self.assertTrue((br[1] == _jr[1]).all())

    def test_jacfwd1(self):
        def f1(x, y):
            r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
            return r

        _x = bm.array([1., 2., 3.])
        _y = bm.array([10., 5.])

        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()
                self.x = bm.Variable(bm.array([1., 2., 3.]))
                self.y = bm.Variable(bm.array([10., 5.]))

            def __call__(self, ):
                a = self.x[0] * self.y[0]
                b = 5 * self.x[2] * self.y[1]
                c = 4 * self.x[1] ** 2 - 2 * self.x[2]
                d = self.x[2] * jnp.sin(self.x[0])
                r = jnp.asarray([a, b, c, d])
                return r

        _jr = jax.jacfwd(f1)(_x, _y)
        t = Test()
        br = bm.jacfwd(t, grad_vars=t.x)()
        self.assertTrue((br == _jr).all())

        _jr = jax.jacfwd(f1, argnums=(0, 1))(_x, _y)
        t = Test()
        br = bm.jacfwd(t, grad_vars=[t.x, t.y])()
        self.assertTrue((br[0] == _jr[0]).all())
        self.assertTrue((br[1] == _jr[1]).all())

    def test_jacrev2(self):
        def f1(x, y):
            r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
            return r

        _x = bm.array([1., 2., 3.])
        _y = bm.array([10., 5.])

        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()
                self.x = bm.Variable(bm.array([1., 2., 3.]))

            def __call__(self, y):
                a = self.x[0] * y[0]
                b = 5 * self.x[2] * y[1]
                c = 4 * self.x[1] ** 2 - 2 * self.x[2]
                d = self.x[2] * jnp.sin(self.x[0])
                r = jnp.asarray([a, b, c, d])
                return r

        _jr = jax.jacrev(f1)(_x, _y)
        t = Test()
        br = bm.jacrev(t, grad_vars=t.x)(_y)
        self.assertTrue((br == _jr).all())

        _jr = jax.jacrev(f1, argnums=(0, 1))(_x, _y)
        t = Test()
        var_grads, arg_grads = bm.jacrev(t, grad_vars=t.x, argnums=0)(_y)
        print(var_grads, )
        print(arg_grads, )
        self.assertTrue((var_grads == _jr[0]).all())
        self.assertTrue((arg_grads == _jr[1]).all())

    def test_jacfwd2(self):
        def f1(x, y):
            r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
            return r

        _x = bm.array([1., 2., 3.])
        _y = bm.array([10., 5.])

        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()
                self.x = bm.Variable(bm.array([1., 2., 3.]))

            def __call__(self, y):
                a = self.x[0] * y[0]
                b = 5 * self.x[2] * y[1]
                c = 4 * self.x[1] ** 2 - 2 * self.x[2]
                d = self.x[2] * jnp.sin(self.x[0])
                r = jnp.asarray([a, b, c, d])
                return r

        _jr = jax.jacfwd(f1)(_x, _y)
        t = Test()
        br = bm.jacfwd(t, grad_vars=t.x)(_y)
        self.assertTrue((br == _jr).all())

        _jr = jax.jacfwd(f1, argnums=(0, 1))(_x, _y)
        t = Test()
        var_grads, arg_grads = bm.jacfwd(t, grad_vars=t.x, argnums=0)(_y)
        print(var_grads, )
        print(arg_grads, )
        self.assertTrue((var_grads == _jr[0]).all())
        self.assertTrue((arg_grads == _jr[1]).all())

    def test_jacrev_aux1(self):
        bm.enable_x64()

        def f1(x, y):
            r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
            return r

        _x = bm.array([1., 2., 3.])
        _y = bm.array([10., 5.])

        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()
                self.x = bm.Variable(bm.array([1., 2., 3.]))

            def __call__(self, y):
                a = self.x[0] * y[0]
                b = 5 * self.x[2] * y[1]
                c = 4 * self.x[1] ** 2 - 2 * self.x[2]
                d = self.x[2] * jnp.sin(self.x[0])
                r = jnp.asarray([a, b, c, d])
                return r, (c, d)

        _jr = jax.jacrev(f1)(_x, _y)
        t = Test()
        br, _ = bm.jacrev(t, grad_vars=t.x, has_aux=True)(_y)
        self.assertTrue((br == _jr).all())

        t = Test()
        _jr = jax.jacrev(f1, argnums=(0, 1))(_x, _y)
        _aux = t(_y)[1]
        (var_grads, arg_grads), aux = bm.jacrev(t, grad_vars=t.x, argnums=0, has_aux=True)(_y)
        print(var_grads, )
        print(arg_grads, )
        self.assertTrue((var_grads == _jr[0]).all())
        self.assertTrue((arg_grads == _jr[1]).all())
        self.assertTrue(bm.array_equal(aux, _aux))

        bm.disable_x64()

    def test_jacfwd_aux1(self):
        bm.enable_x64()

        def f1(x, y):
            r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
            return r

        _x = bm.array([1., 2., 3.])
        _y = bm.array([10., 5.])

        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()
                self.x = bm.Variable(bm.array([1., 2., 3.]))

            def __call__(self, y):
                a = self.x[0] * y[0]
                b = 5 * self.x[2] * y[1]
                c = 4 * self.x[1] ** 2 - 2 * self.x[2]
                d = self.x[2] * jnp.sin(self.x[0])
                r = jnp.asarray([a, b, c, d])
                return r, (c, d)

        _jr = jax.jacfwd(f1)(_x, _y)
        t = Test()
        br, (c, d) = bm.jacfwd(t, grad_vars=t.x, has_aux=True)(_y)
        # print(_jr)
        # print(br)
        a = (br == _jr)
        self.assertTrue(a.all())

        t = Test()
        _jr = jax.jacfwd(f1, argnums=(0, 1))(_x, _y)
        _aux = t(_y)[1]
        (var_grads, arg_grads), aux = bm.jacfwd(t, grad_vars=t.x, argnums=0, has_aux=True)(_y)
        print(var_grads, )
        print(arg_grads, )
        self.assertTrue((var_grads == _jr[0]).all())
        self.assertTrue((arg_grads == _jr[1]).all())
        self.assertTrue(bm.array_equal(aux, _aux))

        bm.disable_x64()

    def test_jacrev_return_aux1(self):
        bm.enable_x64()

        def f1(x, y):
            r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
            return r

        _x = bm.array([1., 2., 3.])
        _y = bm.array([10., 5.])

        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()
                self.x = bm.Variable(bm.array([1., 2., 3.]))

            def __call__(self, y):
                a = self.x[0] * y[0]
                b = 5 * self.x[2] * y[1]
                c = 4 * self.x[1] ** 2 - 2 * self.x[2]
                d = self.x[2] * jnp.sin(self.x[0])
                r = jnp.asarray([a, b, c, d])
                return r, (c, d)

        _jr = jax.jacrev(f1)(_x, _y)
        t = Test()
        br, _ = bm.jacrev(t, grad_vars=t.x, has_aux=True)(_y)
        self.assertTrue((br == _jr).all())

        t = Test()
        _jr = jax.jacrev(f1, argnums=(0, 1))(_x, _y)
        _val, _aux = t(_y)
        (var_grads, arg_grads), value, aux = bm.jacrev(t, grad_vars=t.x, argnums=0, has_aux=True, return_value=True)(_y)
        print(var_grads, )
        print(arg_grads, )
        self.assertTrue((var_grads == _jr[0]).all())
        self.assertTrue((arg_grads == _jr[1]).all())
        self.assertTrue(bm.array_equal(aux, _aux))
        self.assertTrue(bm.array_equal(value, _val))

        bm.disable_x64()

    def test_jacfwd_return_aux1(self):
        bm.enable_x64()

        def f1(x, y):
            r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
            return r

        _x = bm.array([1., 2., 3.])
        _y = bm.array([10., 5.])

        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()
                self.x = bm.Variable(bm.array([1., 2., 3.]))

            def __call__(self, y):
                a = self.x[0] * y[0]
                b = 5 * self.x[2] * y[1]
                c = 4 * self.x[1] ** 2 - 2 * self.x[2]
                d = self.x[2] * jnp.sin(self.x[0])
                r = jnp.asarray([a, b, c, d])
                return r, (c, d)

        _jr = jax.jacfwd(f1)(_x, _y)
        t = Test()
        br, _ = bm.jacfwd(t, grad_vars=t.x, has_aux=True)(_y)
        self.assertTrue((br == _jr).all())

        t = Test()
        _jr = jax.jacfwd(f1, argnums=(0, 1))(_x, _y)
        _val, _aux = t(_y)
        (var_grads, arg_grads), value, aux = bm.jacfwd(t, grad_vars=t.x, argnums=0, has_aux=True, return_value=True)(_y)
        print(_val, )
        print('_aux: ', _aux, 'aux: ', aux)
        print(var_grads, )
        print(arg_grads, )
        self.assertTrue((var_grads == _jr[0]).all())
        self.assertTrue((arg_grads == _jr[1]).all())
        self.assertTrue(bm.array_equal(aux, _aux))
        self.assertTrue(bm.array_equal(value, _val))

        bm.disable_x64()


class TestPureFuncVectorGrad(unittest.TestCase):
    def test1(self):
        f = lambda x: 3 * x ** 2
        _x = bm.ones(10)
        pprint(bm.vector_grad(f, argnums=0)(_x))

    def test2(self):
        def f(x, y):
            dx = x ** 2 + y ** 2 + 10
            return dx

        _x = bm.ones(5)
        _y = bm.ones(5)

        g = bm.vector_grad(f, argnums=0)(_x, _y)
        pprint(g)
        self.assertTrue(bm.array_equal(g, 2 * _x))

        g = bm.vector_grad(f, argnums=(0,))(_x, _y)
        self.assertTrue(bm.array_equal(g[0], 2 * _x))

        g = bm.vector_grad(f, argnums=(0, 1))(_x, _y)
        pprint(g)
        self.assertTrue(bm.array_equal(g[0], 2 * _x))
        self.assertTrue(bm.array_equal(g[1], 2 * _y))

    def test3(self):
        def f(x, y):
            dx = x ** 2 + y ** 2 + 10
            dy = x ** 3 + y ** 3 - 10
            return dx, dy

        _x = bm.ones(5)
        _y = bm.ones(5)

        g = bm.vector_grad(f, argnums=0)(_x, _y)
        # pprint(g)
        self.assertTrue(bm.array_equal(g, 2 * _x + 3 * _x ** 2))

        g = bm.vector_grad(f, argnums=(0,))(_x, _y)
        self.assertTrue(bm.array_equal(g[0], 2 * _x + 3 * _x ** 2))

        g = bm.vector_grad(f, argnums=(0, 1))(_x, _y)
        # pprint(g)
        self.assertTrue(bm.array_equal(g[0], 2 * _x + 3 * _x ** 2))
        self.assertTrue(bm.array_equal(g[1], 2 * _y + 3 * _y ** 2))

    def test4_2d(self):
        def f(x, y):
            dx = x ** 2 + y ** 2 + 10
            return dx

        _x = bm.ones((5, 5))
        _y = bm.ones((5, 5))

        g = bm.vector_grad(f, argnums=0)(_x, _y)
        pprint(g)
        self.assertTrue(bm.array_equal(g, 2 * _x))

        g = bm.vector_grad(f, argnums=(0,))(_x, _y)
        self.assertTrue(bm.array_equal(g[0], 2 * _x))

        g = bm.vector_grad(f, argnums=(0, 1))(_x, _y)
        pprint(g)
        self.assertTrue(bm.array_equal(g[0], 2 * _x))
        self.assertTrue(bm.array_equal(g[1], 2 * _y))

    def test_aux1(self):
        def f(x, y):
            dx = x ** 2 + y ** 2 + 10
            dy = x ** 3 + y ** 3 - 10
            return dx, dy

        _x = bm.ones(5)
        _y = bm.ones(5)

        g, aux = bm.vector_grad(f, has_aux=True)(_x, _y)
        pprint(g, )
        pprint(aux)
        self.assertTrue(bm.array_equal(g, 2 * _x))
        self.assertTrue(bm.array_equal(aux, _x ** 3 + _y ** 3 - 10))

    def test_return1(self):
        def f(x, y):
            dx = x ** 2 + y ** 2 + 10
            return dx

        _x = bm.ones(5)
        _y = bm.ones(5)

        g, value = bm.vector_grad(f, return_value=True)(_x, _y)
        pprint(g, )
        pprint(value)
        self.assertTrue(bm.array_equal(g, 2 * _x))
        self.assertTrue(bm.array_equal(value, _x ** 2 + _y ** 2 + 10))

    def test_return_aux1(self):
        def f(x, y):
            dx = x ** 2 + y ** 2 + 10
            dy = x ** 3 + y ** 3 - 10
            return dx, dy

        _x = bm.ones(5)
        _y = bm.ones(5)

        g, value, aux = bm.vector_grad(f, has_aux=True, return_value=True)(_x, _y)
        print('grad', g)
        print('value', value)
        print('aux', aux)
        self.assertTrue(bm.array_equal(g, 2 * _x))
        self.assertTrue(bm.array_equal(value, _x ** 2 + _y ** 2 + 10))
        self.assertTrue(bm.array_equal(aux, _x ** 3 + _y ** 3 - 10))


class TestClassFuncVectorGrad(unittest.TestCase):
    def test1(self):
        class Test(bp.BrainPyObject):
            def __init__(self):
                super(Test, self).__init__()
                self.x = bm.Variable(bm.ones(5))
                self.y = bm.Variable(bm.ones(5))

            def __call__(self, *args, **kwargs):
                return self.x ** 2 + self.y ** 2 + 10

        t = Test()

        g = bm.vector_grad(t, grad_vars=t.x)()
        self.assertTrue(bm.array_equal(g, 2 * t.x))

        g = bm.vector_grad(t, grad_vars=(t.x,))()
        self.assertTrue(bm.array_equal(g[0], 2 * t.x))

        g = bm.vector_grad(t, grad_vars=(t.x, t.y))()
        self.assertTrue(bm.array_equal(g[0], 2 * t.x))
        self.assertTrue(bm.array_equal(g[1], 2 * t.y))


def vgrad(f, *x):
    y, vjp_fn = jax.vjp(f, *x)
    return vjp_fn(bm.ones(y.shape).value)[0]


class TestDebug(parameterized.TestCase):
    def test_debug1(self):
        a = bm.random.RandomState()

        def f(b):
            print(a.value)
            return a.value + b + a.random()

        f = bm.vector_grad(f, argnums=0)
        f(1.)

        with jax.disable_jit():
            f(1.)

    def test_debug_correctness1(self):
        def test_f():
            a = bm.Variable(bm.ones(2))
            b = bm.Variable(bm.zeros(2))

            def f1(c):
                a.value += 1
                b.value += 10
                return a * b * c

            return a, b, bm.vector_grad(f1, argnums=0)(1.)

        r1 = test_f()
        print(r1)

        with jax.disable_jit():
            r2 = test_f()
            print(r2)
            self.assertTrue(bm.allclose(r1[0], r2[0]))
            self.assertTrue(bm.allclose(r1[1], r2[1]))
            self.assertTrue(bm.allclose(r1[2], r2[2]))

        def f1(c, a, b):
            a += 1
            b += 10
            return a * b * c

        r3 = vgrad(f1, 1., bm.ones(2).value, bm.zeros(2).value)
        self.assertTrue(bm.allclose(r1[2], r3))

    def _bench_f2(self, dd):
        a = bm.Variable(bm.ones(2))
        b = bm.Variable(bm.zeros(2))

        @bm.jit
        def run_fun(d):
            def f1(c):
                a.value += d
                b.value += 10
                return a * b * c

            return a, b, bm.vector_grad(f1, argnums=0)(1.)

        return run_fun(dd)

    # def test_debug_correctness2(self):
    #   r1 = self._bench_f2(1.)
    #   print(r1)
    #
    #   with jax.disable_jit():
    #     r2 = self._bench_f2(1.)
    #     print(r2)
    #
    #   self.assertTrue(bm.allclose(r1[0], r2[0]))
    #   self.assertTrue(bm.allclose(r1[1], r2[1]))
    #   self.assertTrue(bm.allclose(r1[2], r2[2]))

# class TestHessian(unittest.TestCase):
#   def test_hessian5(self):
#     bm.set_mode(bm.training_mode)
#
#     class RNN(bp.DynamicalSystem):
#       def __init__(self, num_in, num_hidden):
#         super(RNN, self).__init__()
#         self.rnn = bp.dyn.RNNCell(num_in, num_hidden, train_state=True)
#         self.out = bp.dnn.Dense(num_hidden, 1)
#
#       def update(self, x):
#         return self.out(self.rnn(x))
#
#     # define the loss function
#     def lossfunc(inputs, targets):
#       runner = bp.DSTrainer(model, progress_bar=False, numpy_mon_after_run=False)
#       predicts = runner.predict(inputs)
#       loss = bp.losses.mean_squared_error(predicts, targets)
#       return loss
#
#     model = RNN(1, 2)
#     data_x = bm.random.rand(1, 1000, 1)
#     data_y = data_x + bm.random.randn(1, 1000, 1)
#
#     bp.reset_state(model, 1)
#     losshess = bm.hessian(lossfunc, grad_vars=model.train_vars())
#     hess_matrix = losshess(data_x, data_y)
#
#     weights = model.train_vars().unique()
#
#     # define the loss function
#     def loss_func_for_jax(weight_vals, inputs, targets):
#       for k, v in weight_vals.items():
#         weights[k].value = v
#       runner = bp.DSTrainer(model, progress_bar=False, numpy_mon_after_run=False)
#       predicts = runner.predict(inputs)
#       loss = bp.losses.mean_squared_error(predicts, targets)
#       return loss
#
#     bp.reset_state(model, 1)
#     jax_hessian = jax.hessian(loss_func_for_jax, argnums=0)({k: v.value for k, v in weights.items()}, data_x, data_y)
#
#     for k, v in hess_matrix.items():
#       for kk, vv in v.items():
#         self.assertTrue(bm.allclose(vv, jax_hessian[k][kk], atol=1e-4))
#
#     bm.clear_buffer_memory()
