# -*- coding: utf-8 -*-


import jax
import unittest
from pprint import pprint

import jax.numpy as jnp
import pytest

import brainpy as bp
import brainpy.math as bm
from brainpy.base import Base


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


class TestObjectFuncGrad(unittest.TestCase):
  def test_grad_ob1(self):
    class Test(Base):
      def __init__(self):
        super(Test, self).__init__()

        self.a = bm.TrainVar(bm.ones(10))
        self.b = bm.TrainVar(bm.random.randn(10))
        self.c = bm.TrainVar(bm.random.uniform(size=10))

      def __call__(self):
        return bm.sum(self.a + self.b + self.c)

    bm.random.seed(0)

    t = Test()
    f_grad = bm.grad(t, grad_vars=t.vars())
    grads = f_grad()
    for g in grads.values(): assert (g == 1.).all()

    t = Test()
    f_grad = bm.grad(t, grad_vars=[t.a, t.b], dyn_vars=t.vars())
    grads = f_grad()
    for g in grads: assert (g == 1.).all()

    t = Test()
    f_grad = bm.grad(t, grad_vars=t.a, dyn_vars=t.vars())
    grads = f_grad()
    assert (grads == 1.).all()

  def test_grad_ob_aux(self):
    class Test(Base):
      def __init__(self):
        super(Test, self).__init__()
        self.a = bm.TrainVar(bm.ones(10))
        self.b = bm.TrainVar(bm.random.randn(10))
        self.c = bm.TrainVar(bm.random.uniform(size=10))

      def __call__(self):
        return bm.sum(self.a + self.b + self.c), (bm.sin(100), bm.exp(0.1))

    bm.random.seed(0)
    t = Test()
    f_grad = bm.grad(t, grad_vars=[t.a, t.b], dyn_vars=t.vars(), has_aux=True)
    grads, aux = f_grad()
    for g in grads: assert (g == 1.).all()
    assert aux[0] == bm.sin(100)
    assert aux[1] == bm.exp(0.1)

    t = Test()
    f_grad = bm.grad(t, grad_vars=t.a, dyn_vars=t.vars(), has_aux=True)
    grads, aux = f_grad()
    assert (grads == 1.).all()
    assert aux[0] == bm.sin(100)
    assert aux[1] == bm.exp(0.1)

  def test_grad_ob_return(self):
    class Test(Base):
      def __init__(self):
        super(Test, self).__init__()
        self.a = bm.TrainVar(bm.ones(10))
        self.b = bm.TrainVar(bm.random.randn(10))
        self.c = bm.TrainVar(bm.random.uniform(size=10))

      def __call__(self):
        return bm.sum(self.a + self.b + self.c)

    bm.random.seed(0)
    t = Test()
    f_grad = bm.grad(t, grad_vars=[t.a, t.b], dyn_vars=t.vars(), return_value=True)
    grads, returns = f_grad()
    for g in grads: assert (g == 1.).all()
    assert returns == t()

    t = Test()
    f_grad = bm.grad(t, grad_vars=t.a, dyn_vars=t.vars(), return_value=True)
    grads, returns = f_grad()
    assert (grads == 1.).all()
    assert returns == t()

  def test_grad_ob_aux_return(self):
    class Test(Base):
      def __init__(self):
        super(Test, self).__init__()
        self.a = bm.TrainVar(bm.ones(10))
        self.b = bm.TrainVar(bm.random.randn(10))
        self.c = bm.TrainVar(bm.random.uniform(size=10))

      def __call__(self):
        return bm.sum(self.a + self.b + self.c), (bm.sin(100), bm.exp(0.1))

    bm.random.seed(0)
    t = Test()
    f_grad = bm.grad(t, grad_vars=[t.a, t.b], dyn_vars=t.vars(),
                     has_aux=True, return_value=True)
    grads, returns, aux = f_grad()
    for g in grads: assert (g == 1.).all()
    assert returns == bm.sum(t.a + t.b + t.c)
    assert aux[0] == bm.sin(100)
    assert aux[1] == bm.exp(0.1)

    t = Test()
    f_grad = bm.grad(t, grad_vars=t.a, dyn_vars=t.vars(),
                     has_aux=True, return_value=True)
    grads, returns, aux = f_grad()
    assert (grads == 1.).all()
    assert returns == bm.sum(t.a + t.b + t.c)
    assert aux[0] == bm.sin(100)
    assert aux[1] == bm.exp(0.1)

  def test_grad_ob_argnums(self):
    class Test(Base):
      def __init__(self):
        super(Test, self).__init__()

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
    f_grad = bm.grad(t, dyn_vars=t.vars(), argnums=0)
    arg_grads = f_grad(bm.random.random(10))
    assert (arg_grads == 2.).all()

    t = Test()
    f_grad = bm.grad(t, dyn_vars=t.vars(), argnums=[0])
    arg_grads = f_grad(bm.random.random(10))
    assert (arg_grads[0] == 2.).all()

  def test_grad_ob_argnums_aux(self):
    class Test(Base):
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
    f_grad = bm.grad(t, dyn_vars=t.vars(), argnums=0, has_aux=True)
    arg_grads, aux = f_grad(bm.random.random(10))
    assert (arg_grads == 2.).all()
    assert aux[0] == bm.sin(100)
    assert aux[1] == bm.exp(0.1)

    t = Test()
    f_grad = bm.grad(t, dyn_vars=t.vars(), argnums=[0], has_aux=True)
    arg_grads, aux = f_grad(bm.random.random(10))
    assert (arg_grads[0] == 2.).all()
    assert aux[0] == bm.sin(100)
    assert aux[1] == bm.exp(0.1)

  def test_grad_ob_argnums_return(self):
    class Test(Base):
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
    f_grad = bm.grad(t, dyn_vars=t.vars(), argnums=0, return_value=True)
    d = bm.random.random(10)
    arg_grads, loss = f_grad(d)
    assert (arg_grads == 2.).all()
    assert loss == t(d)

    t = Test()
    f_grad = bm.grad(t, dyn_vars=t.vars(), argnums=[0], return_value=True)
    d = bm.random.random(10)
    arg_grads, loss = f_grad(d)
    assert (arg_grads[0] == 2.).all()
    assert loss == t(d)

  def test_grad_ob_argnums_aux_return(self):
    class Test(Base):
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
    f_grad = bm.grad(t, dyn_vars=t.vars(), argnums=0, has_aux=True, return_value=True)
    d = bm.random.random(10)
    arg_grads, loss, aux = f_grad(d)
    assert (arg_grads == 2.).all()
    assert aux[0] == bm.sin(100)
    assert aux[1] == bm.exp(0.1)
    assert loss == t(d)[0]

    t = Test()
    f_grad = bm.grad(t, dyn_vars=t.vars(), argnums=[0], has_aux=True, return_value=True)
    d = bm.random.random(10)
    arg_grads, loss, aux = f_grad(d)
    assert (arg_grads[0] == 2.).all()
    assert aux[0] == bm.sin(100)
    assert aux[1] == bm.exp(0.1)
    assert loss == t(d)[0]


class TestPureFuncJacrev(unittest.TestCase):
  def test_jacrev1(self):
    def f1(x, y):
      r = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1],
                       4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
      return r

    br = bm.jacrev(f1)(bm.array([1., 2., 3.]), bm.array([10., 5.]))
    jr = jax.jacrev(f1)(bm.array([1., 2., 3.]), bm.array([10., 5.]))
    assert (br == jr).all()

    br = bm.jacrev(f1, argnums=(0, 1))(bm.array([1., 2., 3.]), bm.array([10., 5.]))
    jr = jax.jacrev(f1, argnums=(0, 1))(bm.array([1., 2., 3.]), bm.array([10., 5.]))
    assert (br[0] == jr[0]).all()
    assert (br[1] == jr[1]).all()

    def f2(x, y):
      r1 = jnp.asarray([x[0] * y[0], 5 * x[2] * y[1]])
      r2 = jnp.asarray([4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
      return r1, r2

    # br = bm.jacrev(f2)(bm.array([1., 2., 3.]), bm.array([10., 5.]))
    # print(br)
    jr = jax.jacrev(f2)(bm.array([1., 2., 3.]), bm.array([10., 5.]))
    print(jr)
    # assert (br == jr).all()

    # def f(x):
    #   r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
    #   return r, 1
    #
    # fr = _jacrev(f, has_aux=True)(jnp.array([1., 2., 3.]))
    # assert len(fr) == 2
    # print(fr)
    #
    # def f(x):
    #   r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
    #   return r, (r, 1)
    #
    # fr = _jacrev(f, has_aux=True)(jnp.array([1., 2., 3.]))
    # assert len(fr) == 2
    # assert len(fr[1]) == 2
    # print(fr)

class TestJacrev(unittest.TestCase):



  def test_jacrev_aux2(self):
    print()

    def f(x):
      r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
      return r

    with pytest.raises(Exception):
      fr = _jacrev(f, has_aux=True)(jnp.array([1., 2., 3.]))

  def test_jacrev_aux3(self):
    print()

    def f(x):
      r1 = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
      r2 = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2]])
      return r1, r2

    fr = _jacrev(f, has_aux=False)(jnp.array([1., 2., 3.]))
    print(fr[0])
    print(fr[1])

  def test_jacrev_aux4(self):
    print()

    def f(x):
      r1 = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
      r2 = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2]])
      return (r1, r2), 1.

    fr = _jacrev(f, has_aux=True)(jnp.array([1., 2., 3.]))
    print(fr[0][0])
    print(fr[0][1])
    print(fr[1])

  def test_jacrev1(self):
    print()

    def f(x):
      r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
      return r

    fr = jacrev(f, has_aux=False)(jnp.array([1., 2., 3.]))
    print(fr)

    print('-' * 20)

    def f(x):
      r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
      return r, 1

    fr = _jacrev(f, has_aux=True)(jnp.array([1., 2., 3.]))
    assert len(fr) == 2
    print(fr)

    print('-' * 20)

    def f(x):
      r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
      return r, (r, 1)

    fr = _jacrev(f, has_aux=True)(jnp.array([1., 2., 3.]))
    assert len(fr) == 2
    assert len(fr[1]) == 2
    pprint(fr)

  def test_jacrev2(self):
    print()

    def f(x, y):
      r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
      r += y
      return r

    fr = jacrev(f)(jnp.array([1., 2., 3.]), jnp.zeros(4))
    # fr = jax.jacrev(f)(jnp.array([1., 2., 3.]), jnp.zeros(4))
    print(fr)

    print('-' * 20)

  def test_jacrev_object1(self):
    print()

    def f(x):
      r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
      return r

    fr1 = jacrev(f)(jnp.array([1., 2., 3.]))

    class F(bp.Base):
      def __init__(self):
        super(F, self).__init__()
        self.x = bm.array([1., 2., 3.])

      def __call__(self, *args, **kwargs):
        r = jnp.asarray([self.x[0], 5 * self.x[2],
                         4 * self.x[1] ** 2 - 2 * self.x[2],
                         self.x[2] * jnp.sin(self.x[0])])
        return r

    f = F()
    fr2 = jacrev(f, grad_vars=f.x)()
    print(fr2)

    assert (fr1 == fr2).all()

    print('-' * 20)

  def test_jacrev_object2(self):
    print()

    class F(bp.Base):
      def __init__(self):
        super(F, self).__init__()
        self.x = bm.array([1., 2., 3.])

      def __call__(self, x2, **kwargs):
        r = jnp.asarray([self.x[0], 5 * self.x[2],
                         4 * self.x[1] ** 2 - 2 * self.x[2],
                         self.x[2] * jnp.sin(self.x[0])])
        r += x2
        return r

    f = F()
    fr2 = jacrev(f, grad_vars=f.x)(bm.random.normal(4).value)
    print(fr2)

    print('-' * 20)


class TestJacfwd(unittest.TestCase):

  def test_jacfwd_aux1(self):
    print()

    def f(x):
      r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
      return r

    fr = _jac_fwd_aux(f, has_aux=False)(jnp.array([1., 2., 3.]))
    print(fr)

  def test_jacfwd_aux2(self):
    print()

    def f(x):
      r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
      return r, jnp.zeros(1)

    fr = _jac_fwd_aux(f, has_aux=True)(jnp.array([1., 2., 3.]))
    print(fr)

  def test_jacfwd_aux3(self):
    print()

    def f(x):
      r1 = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
      r2 = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2]])
      return (r1, r2)

    fr = _jac_fwd_aux(f, has_aux=False)(jnp.array([1., 2., 3.]))
    print(fr[0])
    print(fr[1])
