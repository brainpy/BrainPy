# -*- coding: utf-8 -*-


import unittest

import jax.numpy as jnp

import brainpy.math as bm


class TestSyn2Post(unittest.TestCase):
  def test_syn2post_sum(self):
    data = bm.arange(5)
    segment_ids = bm.array([0, 0, 1, 1, 2])
    self.assertTrue(bm.array_equal(bm.syn2post_sum(data, segment_ids, 3),
                                   bm.asarray([1, 5, 4])))

  def test_syn2post_max(self):
    data = bm.arange(5)
    segment_ids = bm.array([0, 0, 1, 1, 2])
    self.assertTrue(bm.array_equal(bm.syn2post_max(data, segment_ids, 3),
                                   bm.asarray([1, 3, 4])))

  def test_syn2post_min(self):
    data = bm.arange(5)
    segment_ids = bm.array([0, 0, 1, 1, 2])
    self.assertTrue(bm.array_equal(bm.syn2post_min(data, segment_ids, 3),
                                   bm.asarray([0, 2, 4])))

  def test_syn2post_prod(self):
    data = bm.arange(5)
    segment_ids = bm.array([0, 0, 1, 1, 2])
    self.assertTrue(bm.array_equal(bm.syn2post_prod(data, segment_ids, 3),
                                   bm.asarray([0, 6, 4])))

  def test_syn2post_mean(self):
    data = bm.arange(5)
    segment_ids = bm.array([0, 0, 1, 1, 2])
    self.assertTrue(bm.array_equal(bm.syn2post_mean(data, segment_ids, 3),
                                   bm.asarray([0.5, 2.5, 4.])))

  def test_syn2post_softmax(self):
    data = bm.arange(5)
    segment_ids = bm.array([0, 0, 1, 1, 2])
    f_ans = bm.syn2post_softmax(data, segment_ids, 3)
    true_ans = bm.asarray([jnp.exp(data[0]) / (jnp.exp(data[0]) + jnp.exp(data[1])),
                           jnp.exp(data[1]) / (jnp.exp(data[0]) + jnp.exp(data[1])),
                           jnp.exp(data[2]) / (jnp.exp(data[2]) + jnp.exp(data[3])),
                           jnp.exp(data[3]) / (jnp.exp(data[2]) + jnp.exp(data[3])),
                           jnp.exp(data[4]) / jnp.exp(data[4])])
    print()
    print(bm.asarray(f_ans))
    print(true_ans)
    print(f_ans == true_ans)
    # self.assertTrue(bm.array_equal(bm.syn2post_softmax(data, segment_ids, 3),
    #                                true_ans))

    data = bm.arange(5)
    segment_ids = bm.array([0, 0, 1, 1, 2])
    print(bm.syn2post_softmax(data, segment_ids, 4))
