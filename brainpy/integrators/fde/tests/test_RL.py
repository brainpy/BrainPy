# -*- coding: utf-8 -*-

import unittest
from brainpy.integrators.fde.RL import RLmatrix
import brainpy.math as bm


class TestRLAlgorithm(unittest.TestCase):
  def test_RL_matrix_shape(self):
    bm.enable_x64()
    print()
    print(RLmatrix(0.4, 5))
    self.assertTrue(RLmatrix(0.4, 10).shape == (10, 10))
    self.assertTrue(RLmatrix(1.2, 5).shape == (5, 5))


