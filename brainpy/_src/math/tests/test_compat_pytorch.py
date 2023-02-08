# -*- coding: utf-8 -*-




import jax.numpy as jnp
import unittest
import brainpy.math as bm
from brainpy._src.math import compat_pytorch

from absl .testing import parameterized


class TestFlatten(unittest.TestCase):
  def test1(self):
    rng = bm.random.default_rng(113)
    arr = rng.rand(3, 4, 5)
    a2 = compat_pytorch.flatten(arr, 1, 2)
    self.assertTrue(a2.shape == (3, 20))
    a2 = compat_pytorch.flatten(arr, 0, 1)
    self.assertTrue(a2.shape == (12, 5))

  def test2(self):
    rng = bm.random.default_rng(234)
    arr = rng.rand()
    self.assertTrue(arr.ndim == 0)
    arr = compat_pytorch.flatten(arr)
    self.assertTrue(arr.ndim == 1)


class TestUnsqueeze(unittest.TestCase):
  def test1(self):
    rng = bm.random.default_rng(999)
    arr = rng.rand(3, 4, 5)
    a = compat_pytorch.unsqueeze(arr, 0)
    self.assertTrue(a.shape == (1, 3, 4, 5))
    a = compat_pytorch.unsqueeze(arr, -3)
    self.assertTrue(a.shape == (3, 1, 4, 5))

class TestExpand(unittest.TestCase):
  def test1(self):
    rng = bm.random.default_rng(121)
    arr = rng.rand(1, 4, 5)
    a = compat_pytorch.Tensor(arr)
    a = a.expand(1, 6, 4, -1)
    self.assertTrue(a.shape == (1, 6, 4, 5))

