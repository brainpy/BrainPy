# -*- coding: utf-8 -*-


import unittest

import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten

import brainpy.math as bm
from brainpy.math import Variable


class TestJaxArray(unittest.TestCase):
  def test_tree(self):
    structured = {'a': Variable(jnp.zeros(1)),
                  'b': (Variable(jnp.ones(2)),
                        Variable(jnp.ones(2) * 2))}
    flat, tree = tree_flatten(structured)
    unflattened = tree_unflatten(tree, flat)
    print("\nstructured={}\n\n  flat={}\n\n  tree={}\n\n  unflattened={}".format(
      structured, flat, tree, unflattened))

  def test_none(self):
    # https://github.com/PKU-NIP-Lab/BrainPy/issues/144
    a = None
    b = bm.zeros(10)
    with self.assertRaises(TypeError):
      bb = a + b

    c = bm.Variable(bm.zeros(10))
    with self.assertRaises(TypeError):
      cc = a + c

    d = bm.Parameter(bm.zeros(10))
    with self.assertRaises(TypeError):
      dd = a + d

    e = bm.TrainVar(bm.zeros(10))
    with self.assertRaises(TypeError):
      ee = a + e


