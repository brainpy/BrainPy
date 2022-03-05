# -*- coding: utf-8 -*-

import unittest

import brainpy.math as bm


class TestFixedLenDelay(unittest.TestCase):
  def test_dim1(self):
    t0 = 0.
    before_t0 = bm.repeat(bm.arange(10).reshape((-1, 1)), 10, axis=1)
    delay = bm.FixedLenDelay(10, delay_len=1., t0=t0, dt=0.1, before_t0=before_t0)
    self.assertTrue(bm.array_equal(delay(t0 - 0.1), bm.ones(10) * 9))
    self.assertTrue(bm.array_equal(delay(t0 - 0.15), bm.ones(10) * 8.5))
    self.assertTrue(bm.array_equal(delay(t0 - 0.23), bm.ones(10) * 7.7))

  def test_dim2(self):
    t0 = 0.
    before_t0 = bm.repeat(bm.arange(10).reshape((-1, 1)), 10, axis=1)
    before_t0 = bm.repeat(before_t0.reshape((10, 10, 1)), 5, axis=2)
    delay = bm.FixedLenDelay((10, 5), delay_len=1., t0=t0, dt=0.1, before_t0=before_t0)
    self.assertTrue(bm.array_equal(delay(t0 - 0.1), bm.ones((10, 5)) * 9))
    self.assertTrue(bm.array_equal(delay(t0 - 0.15), bm.ones((10, 5)) * 8.5))
    self.assertTrue(bm.array_equal(delay(t0 - 0.23), bm.ones((10, 5)) * 7.7))

  def test_dim3(self):
      t0 = 0.
      before_t0 = bm.repeat(bm.arange(10).reshape((-1, 1)), 10, axis=1)
      before_t0 = bm.repeat(before_t0.reshape((10, 10, 1)), 5, axis=2)
      before_t0 = bm.repeat(before_t0.reshape((10, 10, 5, 1)), 3, axis=3)
      delay = bm.FixedLenDelay((10, 5, 3), delay_len=1., t0=t0, dt=0.1, before_t0=before_t0)
      self.assertTrue(bm.array_equal(delay(t0 - 0.1), bm.ones((10, 5, 3)) * 9))
      self.assertTrue(bm.array_equal(delay(t0 - 0.15), bm.ones((10, 5, 3)) * 8.5))
      self.assertTrue(bm.array_equal(delay(t0 - 0.23), bm.ones((10, 5, 3)) * 7.7))

  def test1(self):
    print()
    delay = bm.FixedLenDelay(3, delay_len=1., dt=0.1, before_t0=lambda t: t)
    print(delay(-0.2))
    delay = bm.FixedLenDelay((3, 2), delay_len=1., dt=0.1, before_t0=lambda t: t)
    print(delay(-0.6))
    delay = bm.FixedLenDelay((3, 2, 1), delay_len=1., dt=0.1, before_t0=lambda t: t)
    print(delay(-0.8))

  def test_current_time2(self):
    print()
    delay = bm.FixedLenDelay(3, delay_len=1., dt=0.1, before_t0=lambda t: t)
    print(delay(0.))
    before_t0 = bm.repeat(bm.arange(10).reshape((-1, 1)), 10, axis=1)
    before_t0 = bm.repeat(before_t0.reshape((10, 10, 1)), 5, axis=2)
    delay = bm.FixedLenDelay((10, 5), delay_len=1., dt=0.1, before_t0=before_t0)
    print(delay(0.))

  def test_prev_time_beyond_boundary(self):
    with self.assertRaises(ValueError):
      delay = bm.FixedLenDelay(3, delay_len=1., dt=0.1, before_t0=lambda t: t)
      delay(-1.1)

