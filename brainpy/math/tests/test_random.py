import unittest

import jax.numpy as jnp
import jax.random as jr
import brainpy.math as bm
import brainpy.math.random as br
import numpy.random as nr

class TestRandom(unittest.TestCase):
  def test_seed(self):
    test_seed = 299
    br.seed(test_seed)
    a = br.rand(3)
    br.seed(test_seed)
    b = br.rand(3)
    self.assertTrue(bm.array_equal(a, b))

  def test_rand(self):
    a = br.rand(3, 2)
    self.assertTupleEqual(a.shape, (3, 2))
    self.assertTrue((a >= 0).all() and (a < 1).all())

  def test_randint1(self):
    a = br.randint(5, size=10)
    self.assertTupleEqual(a.shape, (10,))
    self.assertTrue((a >= 0).all() and (a < 5).all())

  def test_randint2(self):
    a = br.randint(2, 6, size=(4, 3))
    self.assertTupleEqual(a.shape, (4, 3))
    self.assertTrue((a >= 2).all() and (a < 6).all())

  # def test_randint3(self):
  #   a = br.randint([1, 2, 3], [10, 7, 8], size=3)
  #   self.assertTupleEqual(a.shape, (3,))
  #   self.assertTrue((a - bm.array([1, 2, 3]) >= 0).all()
  #                   and (-a + bm.array([10, 7, 8]) > 0).all())

  def test_randn(self):
    a = br.randn(3, 2)
    self.assertTupleEqual(a.shape, (3, 2))

  def test_random1(self):
    a = br.random()
    self.assertIsInstance(a, bm.jaxarray.JaxArray)
    self.assertTrue(0. <= a < 1)

  def test_random2(self):
    a = br.random(size=(3, 2))
    self.assertTupleEqual(a.shape, (3, 2))
    self.assertTrue((a >= 0).all() and (a < 1).all())

  def test_random_sample(self):
    a = br.random_sample(size=(3, 2))
    self.assertTupleEqual(a.shape, (3, 2))
    self.assertTrue((a >= 0).all() and (a < 1).all())