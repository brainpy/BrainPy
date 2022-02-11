# -*- coding: utf-8 -*-

import unittest

import jax.numpy as jnp

import brainpy.math.numpy_ops as bm


class TestNumPyOPS(unittest.TestCase):
  def test_asarray1(self):
    a = [bm.zeros(1), bm.ones(1)]
    print(bm.asarray(a))
    self.assertTrue(bm.array_equal(bm.asarray(a), bm.array([[0.], [1.]])))
    self.assertTrue(bm.array_equal(bm.array(a), bm.array([[0.], [1.]])))

  def test_asarray2(self):
    a = [jnp.zeros(1), jnp.ones(1)]
    print(bm.asarray(a))
    self.assertTrue(bm.array_equal(bm.asarray(a), bm.array([[0.], [1.]])))
    self.assertTrue(bm.array_equal(bm.array(a), bm.array([[0.], [1.]])))

  def test_asarray3(self):
    a = [[0], bm.ones(1)]
    print(bm.asarray(a))
    self.assertTrue(bm.array_equal(bm.asarray(a), bm.array([[0.], [1.]])))
    self.assertTrue(bm.array_equal(bm.array(a), bm.array([[0.], [1.]])))
