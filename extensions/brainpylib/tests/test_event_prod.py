# -*- coding: utf-8 -*-

import unittest

from brainpylib import event_prod

import brainpy as bp
import brainpy.math as bm

# bm.set_platform('gpu')


class TestEventProd(unittest.TestCase):
  def test_homo_values(self):
    bp.math.random.seed(1345)
    size = 200
    conn = bp.conn.FixedProb(prob=0.5, seed=123)
    # conn = bp.conn.All2All()
    conn(pre_size=size, post_size=size)
    post_ids, indptr = conn.require('pre2post')
    sps = bm.random.random(size).value < 0.5
    # print(sps)
    value = 1.0233
    a = event_prod(sps, (post_ids.value, indptr.value), size, value)
    print(a)

  def test_heter_value(self):
    bp.math.random.seed(3)
    size = 200
    conn = bp.conn.FixedProb(prob=0.5, seed=3)
    # conn = bp.conn.One2One()
    conn(pre_size=size, post_size=size)
    post_ids, indptr = conn.require('pre2post')
    # sps = bm.random.randint(0, 2, size).value < 1
    sps = bm.random.random(size).value < 0.5
    values = bm.random.rand(post_ids.size)
    # values = bm.ones(post_ids.size)
    a = event_prod(sps, (post_ids.value, indptr.value), size, values.value)
    print(a)

