# -*- coding: utf-8 -*-

import brainpy as bp
from brainpy.math import jax as jnp
from brainpy.primary import Primary

bp.math.use_backend('jax')


class Obj1(Primary):
  def __init__(self):
    super(Obj1, self).__init__()

    self.a = jnp.ones(10)
    self.b = jnp.random.randn(10)
    self.c = jnp.random.uniform(10)

  def sum(self):
    return jnp.sum(self.a + self.b + self.c)




def test1():
  class Obj2(Primary):
    def __init__(self):
      super(Obj2, self).__init__()

      self.a = jnp.ones(10)
      self.b = jnp.random.randn(10)
      self.c = jnp.random.uniform(size=10)

    def __call__(self):
      return jnp.sum(self.a + self.b + self.c)

  o2 = Obj2()
  o2_grad = jnp.grad(o2, o2.vars())

  print(o2_grad())
  for g in o2_grad():
    assert (g == 1.).all()


def test2():
  class Obj2(Primary):
    def __init__(self):
      super(Obj2, self).__init__()

      self.a = jnp.ones(10)
      self.b = jnp.random.randn(10)
      self.c = jnp.random.uniform(size=10)

    def __call__(self, d):
      return jnp.sum(self.a + self.b + self.c + d)

  o2 = Obj2()
  o2_grad = jnp.grad(o2, o2.vars())

  res = o2_grad(jnp.random.random(10))
  print(res)
  for g in res:
    assert (g == 1.).all()


def test3():
  class Obj2(Primary):
    def __init__(self):
      super(Obj2, self).__init__()

      self.a = jnp.ones(10)
      self.b = jnp.random.randn(10)
      self.c = jnp.random.uniform(size=10)

    def __call__(self, d):
      return jnp.sum(self.a + self.b + self.c + 2 * d)

  o2 = Obj2()
  o2_grad = jnp.grad(o2, o2.vars(), argnums=0)

  res = o2_grad(jnp.random.random(10))
  print(res)
  for g in res[1]:
    assert (g == 1.).all()
  assert (res[0] == 2.).all()


