# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
import unittest


class TestSliceView(unittest.TestCase):
  def test_lif(self):
    lif = bp.neurons.LIF(10)
    lif_tile = lif[5:]
    print(lif_tile.V.shape)
    print(lif_tile.varshape)

    print('Before modification: ')
    print(lif.V)
    lif_tile.V += 10.

    self.assertTrue(bm.allclose(lif.V, bm.concatenate([bm.zeros(5), bm.ones(5) * 10.])))
    print('After modification 1: ')
    print(lif.V)

    lif_tile.V.value = bm.ones(5) * 40.
    self.assertTrue(bm.allclose(lif.V, bm.concatenate([bm.zeros(5), bm.ones(5) * 40.])))
    print('After modification 2: ')
    print(lif.V)

  def test_lif_train_mode(self):
    lif = bp.neurons.LIF(10, mode=bp.modes.training)
    lif_tile = lif[5:]
    print(lif_tile.V.shape)
    print(lif_tile.varshape)

    print('Before modification: ')
    print(lif.V)
    lif_tile.V += 10.

    self.assertTrue(bm.allclose(lif.V, bm.hstack([bm.zeros((1, 5)), bm.ones((1, 5)) * 10.])))
    print('After modification 1: ')
    print(lif.V)

    lif_tile.V.value = bm.ones((1, 5)) * 40.
    self.assertTrue(bm.allclose(lif.V, bm.hstack([bm.zeros((1, 5)), bm.ones((1, 5)) * 40.])))
    print('After modification 2: ')
    print(lif.V)





