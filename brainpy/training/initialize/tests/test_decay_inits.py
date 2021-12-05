# -*- coding: utf-8 -*-
import unittest

import matplotlib.pyplot as plt

import brainpy as bp

PI = bp.math.pi


def _size2len(size):
  if isinstance(size, int):
    return size
  elif isinstance(size, (tuple, list)):
    length = 1
    for e in size:
      length *= e
    return length
  else:
    raise ValueError(f'Must be a list/tuple of int, but got {size}')


class TestGaussianDecayInit(unittest.TestCase):
  def test_gaussian_decay_init1(self):
    init = bp.init.GaussianDecay(sigma=4, max_w=1.)
    for size in [10, (10, 20), (10, 20, 30)]:
      weights = init(size)
      shape = _size2len(size)
      assert weights.shape == (shape, shape)
      assert isinstance(weights, bp.math.ndarray)
      # plt.imshow(weights)
      # plt.show()

  def test_gaussian_decay_init2(self):
    init = bp.init.GaussianDecay(sigma=4, max_w=1., min_w=0.1, periodic_boundary=True,
                                 encoding_values=((-PI, PI), (10, 20), (0, 2 * PI)),
                                 include_self=False, normalize=True)
    size = (10, 20, 30)
    weights = init(size)
    shape = _size2len(size)
    assert weights.shape == (shape, shape)
    assert isinstance(weights, bp.math.ndarray)
    # plt.imshow(weights)
    # plt.show()


class TestDOGDecayInit(unittest.TestCase):
  def test_dog_decay_init1(self):
    init = bp.init.DOGDecay(sigmas=(1., 2.5), max_ws=(1.0, 0.7))
    for size in [10, (10, 20), (10, 20, 30)]:
      weights = init(size)
      shape = _size2len(size)
      assert weights.shape == (shape, shape)
      assert isinstance(weights, bp.math.ndarray)
      # plt.imshow(weights)
      # plt.show()

  def test_dog_decay_init2(self):
    init = bp.init.DOGDecay(sigmas=(1., 2.5),
                            max_ws=(1.0, 0.7), min_w=0.1,
                            periodic_boundary=True,
                            encoding_values=((-PI, PI), (10, 20), (0, 2 * PI)),
                            include_self=False,
                            normalize=True)
    size = (10, 20, 30)
    weights = init(size)
    shape = _size2len(size)
    assert weights.shape == (shape, shape)
    assert isinstance(weights, bp.math.ndarray)
    # plt.imshow(weights)
    # plt.show()