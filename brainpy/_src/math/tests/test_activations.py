# -*- coding: utf-8 -*-

import unittest
import brainpy.math as bm


class TestActivations(unittest.TestCase):
  def test_get(self):
    print()
    print('identity', bm.activations.get('identity'))
    print('tanh', bm.activations.get('tanh'))
    for n in bm.activations.__all__:
      print(n, bm.activations.get(n))

