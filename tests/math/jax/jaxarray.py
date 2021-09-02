# -*- coding: utf-8 -*-


import jax
import jax.numpy as jnp

from brainpy.math.jax import Variable


def test_varaible1():
  @jax.jit
  def try_variable1(a, b):
    return a + b


  va = Variable(jnp.zeros(10), replicate=lambda *args: jnp.zeros(10))
  vb = Variable(jnp.ones(10), replicate=lambda *args: jnp.ones(10))
  va.duplicate(10)
  vb.duplicate(10)
  print(try_variable1(va, vb))

