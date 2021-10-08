# -*- coding: utf-8 -*-

import brainpy as bp
from brainpy.base import Base

bp.math.use_backend('jax')


def test_grad1():
  class Test(Base):
    def __init__(self):
      super(Test, self).__init__()

      self.a = bp.math.TrainVar(bp.math.ones(10))
      self.b = bp.math.TrainVar(bp.math.random.randn(10))
      self.c = bp.math.TrainVar(bp.math.random.uniform(size=10))

    def __call__(self):
      return bp.math.sum(self.a + self.b + self.c)

  o2 = Test()
  o2_grad = bp.math.grad(o2, o2.vars())
  grads = o2_grad()

  print('test_grad1:')
  print(grads)
  for g in grads.values():
    assert (g == 1.).all()


def test_grad2():
  class Test(Base):
    def __init__(self):
      super(Test, self).__init__()

      self.a = bp.math.TrainVar(bp.math.ones(10))
      self.b = bp.math.TrainVar(bp.math.random.randn(10))
      self.c = bp.math.TrainVar(bp.math.random.uniform(size=10))

    def __call__(self, d):
      return bp.math.sum(self.a + self.b + self.c + d)

  o2 = Test()
  o2_grad = bp.math.grad(o2, o2.vars())

  res = o2_grad(bp.math.random.random(10))
  print('test_grad2:')
  print(res)
  for g in res.values():
    assert (g == 1.).all()


def test_grad3():
  class Test(Base):
    def __init__(self):
      super(Test, self).__init__()

      self.a = bp.math.TrainVar(bp.math.ones(10))
      self.b = bp.math.TrainVar(bp.math.random.randn(10))
      self.c = bp.math.TrainVar(bp.math.random.uniform(size=10))

    def __call__(self, d):
      return bp.math.sum(self.a + self.b + self.c + 2 * d)

  o2 = Test()
  o2_grad = bp.math.grad(o2, o2.vars(), argnums=0)

  res = o2_grad(bp.math.random.random(10))
  print('test_grad3:')
  print(res)
  for g in res[1].values():
    assert (g == 1.).all()
  assert (res[0] == 2.).all()


def test_grad_aux1():
  class Test(bp.dnn.Module):

    def __init__(self):
      super(Test, self).__init__()

      self.a = bp.math.TrainVar(bp.math.ones(1))
      self.b = bp.math.TrainVar(bp.math.ones(1))

    def __call__(self, c):
      ab = self.a * self.b
      ab2 = ab * 2
      vv = ab2 + c
      return vv.sum(), (ab, ab2)

  test = Test()
  test_grad = bp.math.grad(test, test.vars(), argnums=0, has_aux=True)
  grads, outputs = test_grad(10.)
  print('test_grad_aux1:')
  print(grads)
  print(outputs)

  assert (grads[0] == 1.).all()  # grad of 'c'
  for g in grads[1].values():  # grad of TrainVar
    assert (g == 2.).all()
  assert (outputs[0] == 1.).all()  # 'ab'
  assert (outputs[1] == 2.).all()  # 'ab2'


def test_value_and_grad_aux1():
  class Test(bp.dnn.Module):

    def __init__(self):
      super(Test, self).__init__()

      self.a = bp.math.TrainVar(bp.math.ones(1))
      self.b = bp.math.TrainVar(bp.math.ones(1))

    def __call__(self, c):
      ab = self.a * self.b
      ab2 = ab * 2
      vv = ab2 + c
      return vv.sum(), (ab, ab2)

  test = Test()
  test_grad = bp.math.value_and_grad(test, test.vars(), argnums=0, has_aux=True)
  outputs, grads  = test_grad(10.)
  print('test_value_and_grad_aux1:')
  print(grads)
  print(outputs)

  assert (grads[0] == 1.).all()  # grad of 'c'
  for g in grads[1].values():  # grad of TrainVar
    assert (g == 2.).all()
  assert (outputs[0] == 12.).all()  # 'vv.sum()'
  assert (outputs[1][0] == 1.).all()  # 'ab'
  assert (outputs[1][1] == 2.).all()  # 'ab2'
