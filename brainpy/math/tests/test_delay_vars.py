# -*- coding: utf-8 -*-

import unittest

import brainpy.math as bm


class TestFixedLenDelay(unittest.TestCase):
  def test_dim1(self):
    bm.enable_x64()

    # linear interp
    t0 = 0.
    before_t0 = bm.repeat(bm.arange(11).reshape((-1, 1)), 10, axis=1)
    delay = bm.FixedLenDelay(10, delay_len=1., t0=t0, dt=0.1, before_t0=before_t0)
    self.assertTrue(bm.array_equal(delay(t0 - 0.1), bm.ones(10) * 10))
    self.assertTrue(bm.array_equal(delay(t0 - 0.15), bm.ones(10) * 9.5))
    print()
    print(delay(t0 - 0.23))
    print(delay(t0 - 0.23) - bm.ones(10) * 8.7)
    # self.assertTrue(bm.array_equal(delay(t0 - 0.23), bm.ones(10) * 8.7))

    # round interp
    delay = bm.FixedLenDelay(10, delay_len=1., t0=t0, dt=0.1, before_t0=before_t0,
                             interp_method='round')
    self.assertTrue(bm.array_equal(delay(t0 - 0.1), bm.ones(10) * 10))
    self.assertTrue(bm.array_equal(delay(t0 - 0.15), bm.ones(10) * 10))
    self.assertTrue(bm.array_equal(delay(t0 - 0.2), bm.ones(10) * 9))

  def test_dim2(self):
    t0 = 0.
    before_t0 = bm.repeat(bm.arange(11).reshape((-1, 1)), 10, axis=1)
    before_t0 = bm.repeat(before_t0.reshape((11, 10, 1)), 5, axis=2)
    delay = bm.FixedLenDelay((10, 5), delay_len=1., t0=t0, dt=0.1, before_t0=before_t0)
    self.assertTrue(bm.array_equal(delay(t0 - 0.1), bm.ones((10, 5)) * 10))
    self.assertTrue(bm.array_equal(delay(t0 - 0.15), bm.ones((10, 5)) * 9.5))
    # self.assertTrue(bm.array_equal(delay(t0 - 0.23), bm.ones((10, 5)) * 8.7))

  def test_dim3(self):
      t0 = 0.
      before_t0 = bm.repeat(bm.arange(11).reshape((-1, 1)), 10, axis=1)
      before_t0 = bm.repeat(before_t0.reshape((11, 10, 1)), 5, axis=2)
      before_t0 = bm.repeat(before_t0.reshape((11, 10, 5, 1)), 3, axis=3)
      delay = bm.FixedLenDelay((10, 5, 3), delay_len=1., t0=t0, dt=0.1, before_t0=before_t0)
      self.assertTrue(bm.array_equal(delay(t0 - 0.1), bm.ones((10, 5, 3)) * 10))
      self.assertTrue(bm.array_equal(delay(t0 - 0.15), bm.ones((10, 5, 3)) * 9.5))
      # self.assertTrue(bm.array_equal(delay(t0 - 0.23), bm.ones((10, 5, 3)) * 8.7))

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
    before_t0 = bm.repeat(bm.arange(11).reshape((-1, 1)), 10, axis=1)
    before_t0 = bm.repeat(before_t0.reshape((11, 10, 1)), 5, axis=2)
    delay = bm.FixedLenDelay((10, 5), delay_len=1., dt=0.1, before_t0=before_t0)
    print(delay(0.))

  # def test_prev_time_beyond_boundary(self):
  #   with self.assertRaises(ValueError):
  #     delay = bm.FixedLenDelay(3, delay_len=1., dt=0.1, before_t0=lambda t: t)
  #     delay(-1.2)

