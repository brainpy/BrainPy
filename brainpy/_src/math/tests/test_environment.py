import unittest

import jax

import brainpy.math as bm


class TestEnvironment(unittest.TestCase):
  def test_numpy_func_return(self):
    with bm.environment(numpy_func_return='jax_array'):
      a = bm.random.randn(3, 3)
      self.assertTrue(isinstance(a, jax.Array))
    with bm.environment(numpy_func_return='bp_array'):
      a = bm.random.randn(3, 3)
      self.assertTrue(isinstance(a, bm.Array))
