# -*- coding: utf-8 -*-
import unittest

import brainpy as bp


class TestZeroInit(unittest.TestCase):
  def test_zero_init(self):
      init = bp.init.ZeroInit()
      for size in [(100,), (10, 20), (10, 20, 30)]:
        weights = init(size)
        assert weights.shape == size
        assert isinstance(weights, bp.math.ndarray)


class TestOneInit(unittest.TestCase):
  def test_one_init(self):
      for size in [(100,), (10, 20), (10, 20, 30)]:
        for value in [0., 1., -1.]:
          init = bp.init.OneInit(value=value)
          weights = init(size)
          assert weights.shape == size
          assert (weights == value).all()


class TestIdentityInit(unittest.TestCase):
  def test_identity_init(self):
      for size in [(100,), (10, 20)]:
        for value in [0., 1., -1.]:
          init = bp.init.Identity(value=value)
          weights = init(size)
          if len(size) == 1:
            assert weights.shape == (size[0], size[0])
          else:
            assert weights.shape == size
          assert isinstance(weights, bp.math.ndarray)
