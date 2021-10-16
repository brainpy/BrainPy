# -*- coding: utf-8 -*-


from pprint import pprint

import jax.numpy as jnp
import pytest

import brainpy as bp
import brainpy.math.jax as bm
from brainpy.base import Base
from brainpy.math.jax.gradient import _jac_rev_aux, _jac_fwd_aux
from brainpy.math.jax.gradient import jacrev

bp.math.use_backend('jax')


def test_grad_func1():
  def call(a, b, c):
    return bm.sum(a + b + c)

  grad = bm.grad(call, argnums=[0, 1, 2])
  a = bm.ones(10)
  b = bm.random.randn(10)
  c = bm.random.uniform(size=10)
  grads = grad(a, b, c)

  print('test_grad_ob1:')
  print(grads)
  for g in grads:
    assert (g == 1.).all()


def test_grad_ob1():
  class Test(Base):
    def __init__(self):
      super(Test, self).__init__()

      self.a = bm.TrainVar(bm.ones(10))
      self.b = bm.TrainVar(bm.random.randn(10))
      self.c = bm.TrainVar(bm.random.uniform(size=10))

    def __call__(self):
      return bm.sum(self.a + self.b + self.c)

  t = Test()
  grad = bm.grad(t, t.vars())
  grads = grad()

  print('test_grad_ob1:')
  print(grads)
  for g in grads.values():
    assert (g == 1.).all()


def test_grad_ob2():
  class Test(Base):
    def __init__(self):
      super(Test, self).__init__()

      self.a = bm.TrainVar(bm.ones(10))
      self.b = bm.TrainVar(bm.random.randn(10))
      self.c = bm.TrainVar(bm.random.uniform(size=10))

    def __call__(self):
      return bm.sum(self.a + self.b + self.c)

  t = Test()
  grad = bm.grad(t)
  grads = grad()

  print('test_grad_ob1:')
  pprint(grads)
  for g in grads.values():
    assert (g == 1.).all()


def test_grad_ob3():
  class Test(Base):
    def __init__(self):
      super(Test, self).__init__()

      self.a = bm.TrainVar(bm.ones(10))
      self.b = bm.TrainVar(bm.random.randn(10))
      self.c = bm.TrainVar(bm.random.uniform(size=10))

    def __call__(self):
      return bm.sum(self.a + self.b + self.c)

  t = Test()
  grad = bm.grad(t, grad_vars=[t.a, t.b])
  grads = grad()
  print('test_grad_ob1:')
  pprint(grads)
  for g in grads:
    assert (g == 1.).all()

  print('-' * 30)

  t = Test()
  grad = bm.grad(t, grad_vars=t.a)
  grads = grad()
  print('test_grad_ob1:')
  pprint(grads)
  for g in grads:
    assert (g == 1.).all()


def test_grad_ob4():
  class Test(Base):
    def __init__(self):
      super(Test, self).__init__()

      self.a = bm.TrainVar(bm.ones(10))
      self.b = bm.TrainVar(bm.random.randn(10))
      self.c = bm.TrainVar(bm.random.uniform(size=10))

    def __call__(self, d):
      return bm.sum(self.a + self.b + self.c + d)

  t = Test()
  grad = bm.grad(t, t.vars())

  res = grad(bm.random.random(10))
  print('test_grad_ob4:')
  pprint(res)
  for g in res.values():
    assert (g == 1.).all()


def test_grad_ob5():
  class Test(Base):
    def __init__(self):
      super(Test, self).__init__()

      self.a = bm.TrainVar(bm.ones(10))
      self.b = bm.TrainVar(bm.random.randn(10))
      self.c = bm.TrainVar(bm.random.uniform(size=10))

    def __call__(self, d):
      return bm.sum(self.a + self.b + self.c + 2 * d)

  t = Test()
  grad = bm.grad(t, t.vars(), argnums=0)

  res = grad(bm.random.random(10))
  print('test_grad_ob5:')
  pprint(res)
  for g in res[1].values():
    assert (g == 1.).all()
  assert (res[0] == 2.).all()


def test_grad_ob_aux1():
  class Test(bp.Base):

    def __init__(self):
      super(Test, self).__init__()

      self.a = bm.TrainVar(bm.ones(1))
      self.b = bm.TrainVar(bm.ones(1))

    def __call__(self, c):
      ab = self.a * self.b
      ab2 = ab * 2
      vv = ab2 + c
      return vv.sum(), (ab, ab2)

  test = Test()
  test_grad = bm.grad(test, test.vars(), argnums=0, has_aux=True)
  grads, outputs = test_grad(10.)
  print('test_grad_ob_aux1:')
  pprint(grads)
  pprint(outputs)

  assert (grads[0] == 1.).all()  # grad of 'c'
  for g in grads[1].values():  # grad of TrainVar
    assert (g == 2.).all()
  assert (outputs[0] == 1.).all()  # 'ab'
  assert (outputs[1] == 2.).all()  # 'ab2'


def test_value_and_grad_ob_aux1():
  class Test(bp.Base):

    def __init__(self):
      super(Test, self).__init__()

      self.a = bm.TrainVar(bm.ones(1))
      self.b = bm.TrainVar(bm.ones(1))

    def __call__(self, c):
      ab = self.a * self.b
      ab2 = ab * 2
      vv = ab2 + c
      return vv.sum(), (ab, ab2)

  test = Test()
  test_grad = bm.grad(test, test.vars(), argnums=0, has_aux=True, return_value=True)
  grads, outputs = test_grad(10.)
  print('test_value_and_grad_ob_aux1:')
  print(grads)
  print(outputs)

  assert (grads[0] == 1.).all()  # grad of 'c'
  for g in grads[1].values():  # grad of TrainVar
    assert (g == 2.).all()
  assert (outputs[0] == 12.).all()  # 'vv.sum()'
  assert (outputs[1][0] == 1.).all()  # 'ab'
  assert (outputs[1][1] == 2.).all()  # 'ab2'


def test_jacrev_aux1():
  print()

  def f(x):
    r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
    return r

  fr = _jac_rev_aux(f, has_aux=False)(jnp.array([1., 2., 3.]))
  assert isinstance(fr, jnp.DeviceArray)
  print(fr)

  def f(x):
    r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
    return r, 1

  fr = _jac_rev_aux(f, has_aux=True)(jnp.array([1., 2., 3.]))
  assert len(fr) == 2
  print(fr)

  def f(x):
    r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
    return r, (r, 1)

  fr = _jac_rev_aux(f, has_aux=True)(jnp.array([1., 2., 3.]))
  assert len(fr) == 2
  assert len(fr[1]) == 2
  print(fr)


def test_jacrev_aux2():
  print()

  def f(x):
    r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
    return r

  with pytest.raises(Exception):
    fr = _jac_rev_aux(f, has_aux=True)(jnp.array([1., 2., 3.]))


def test_jacrev_aux3():
  print()

  def f(x):
    r1 = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
    r2 = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2]])
    return r1, r2

  fr = _jac_rev_aux(f, has_aux=False)(jnp.array([1., 2., 3.]))
  print(fr[0])
  print(fr[1])


def test_jacrev_aux4():
  print()

  def f(x):
    r1 = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
    r2 = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2]])
    return (r1, r2), 1.

  fr = _jac_rev_aux(f, has_aux=True)(jnp.array([1., 2., 3.]))
  print(fr[0][0])
  print(fr[0][1])
  print(fr[1])


def test_jacfwd_aux1():
  print()

  def f(x):
    r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
    return r

  fr = _jac_fwd_aux(f, has_aux=False)(jnp.array([1., 2., 3.]))
  print(fr)


def test_jacfwd_aux2():
  print()

  def f(x):
    r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
    return r, jnp.zeros(1)

  fr = _jac_fwd_aux(f, has_aux=True)(jnp.array([1., 2., 3.]))
  print(fr)


def test_jacfwd_aux3():
  print()

  def f(x):
    r1 = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
    r2 = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2]])
    return (r1, r2)

  fr = _jac_fwd_aux(f, has_aux=False)(jnp.array([1., 2., 3.]))
  print(fr[0])
  print(fr[1])


def test_jacrev1():
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

  fr = _jac_rev_aux(f, has_aux=True)(jnp.array([1., 2., 3.]))
  assert len(fr) == 2
  print(fr)

  print('-' * 20)

  def f(x):
    r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
    return r, (r, 1)

  fr = _jac_rev_aux(f, has_aux=True)(jnp.array([1., 2., 3.]))
  assert len(fr) == 2
  assert len(fr[1]) == 2
  pprint(fr)

import jax


def test_jacrev2():
  print()

  def f(x, y):
    r = jnp.asarray([x[0], 5 * x[2], 4 * x[1] ** 2 - 2 * x[2], x[2] * jnp.sin(x[0])])
    r += y
    return r

  fr = jacrev(f)(jnp.array([1., 2., 3.]), jnp.zeros(4))
  # fr = jax.jacrev(f)(jnp.array([1., 2., 3.]), jnp.zeros(4))
  print(fr)

  print('-' * 20)


def test_jacrev_object1():
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


def test_jacrev_object2():
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