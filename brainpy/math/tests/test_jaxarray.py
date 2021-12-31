# -*- coding: utf-8 -*-


import jax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten

from brainpy.math import Variable


def test_tree():
  structured = {'a': Variable(jnp.zeros(1)),
                'b': (Variable(jnp.ones(2)), Variable(jnp.ones(2) * 2))}
  flat, tree = tree_flatten(structured)
  unflattened = tree_unflatten(tree, flat)
  print("\nstructured={}\n\n  flat={}\n\n  tree={}\n\n  unflattened={}".format(
    structured, flat, tree, unflattened))

