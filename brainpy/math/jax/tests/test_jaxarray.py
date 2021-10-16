# -*- coding: utf-8 -*-


import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten

from brainpy.math.jax import Variable


def test_tree():
  structured = {'a': Variable(jnp.zeros(1)),
                'b': (Variable(jnp.ones(2)), Variable(jnp.ones(2) * 2))}
  flat, tree = tree_flatten(structured)
  unflattened = tree_unflatten(tree, flat)
  print("\nstructured={}\n\n  flat={}\n\n  tree={}\n\n  unflattened={}".format(
    structured, flat, tree, unflattened))


def test_varaible1():
  @jax.jit
  def try_variable1(a, b):
    return a + b

  va = Variable(jnp.zeros(10), replicate=lambda *args: jnp.zeros(10))
  vb = Variable(jnp.ones(10), replicate=lambda *args: jnp.ones(10))
  va.duplicate(10)
  vb.duplicate(10)
  print(try_variable1(va, vb))
