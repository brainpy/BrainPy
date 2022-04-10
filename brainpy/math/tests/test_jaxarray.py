# -*- coding: utf-8 -*-


import unittest

import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten

import brainpy.math as bm
from brainpy.errors import MathError
from brainpy.math import Variable


class TestJaxArray(unittest.TestCase):
  def test_tree(self):
    structured = {'a': Variable(jnp.zeros(1)),
                  'b': (Variable(jnp.ones(2)), Variable(jnp.ones(2) * 2))}
    flat, tree = tree_flatten(structured)
    unflattened = tree_unflatten(tree, flat)
    print("\nstructured={}\n\n  flat={}\n\n  tree={}\n\n  unflattened={}".format(
      structured, flat, tree, unflattened))

