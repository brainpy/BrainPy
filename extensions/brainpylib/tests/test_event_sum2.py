# -*- coding: utf-8 -*-

import timeit
import time
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import unittest
from brainpylib import event_sum
import brainpy as bp
import brainpy.math as bm


bm.set_platform('gpu')


class TestEventSum(unittest.TestCase):
  def test_homo_values(self):
    bp.math.random.seed(1345)
    size = 200
    conn = bp.conn.FixedProb(prob=0.5, seed=123)
    # conn = bp.conn.All2All()
    conn(pre_size=size, post_size=size)
    post_ids, indptr = conn.require('pre2post')
    sps = bm.random.random(size).value < 0.5
    # print(sps)
    value = 3.0233
    a = event_sum(sps, (post_ids.value, indptr.value), size, value)
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
    a = event_sum(sps, (post_ids.value, indptr.value), size, values.value)
    print(a)

#
#
# bp.math.random.seed(1345)
# size = 200
# conn = bp.conn.FixedProb(prob=0.5, seed=123)
# conn(pre_size=size, post_size=size)
# post_ids, indptr = conn.require('pre2post')
# sps = bm.random.random(size).value < 0.5
# # print(sps)
# value = 3.0233
# a = event_sum3(sps, (post_ids.value, indptr.value), size, value,
#                 max_post_conn=bm.diff(indptr).max())
# print(a)


# def test1():
# bm.random.seed(123)
# size = 3000
# conn = bp.conn.FixedProb(prob=0.5, seed=123)
# conn(pre_size=size, post_size=size)
# # pre2post = conn.require('pre2post')
# pre_ids, post_ids = conn.require('pre_ids', 'post_ids')
# print("pre_ids size:", pre_ids.size)
# # indices = jnp.arange(size, dtype=jnp.uint32)
# # idnptr = jnp.arange(size + 1, dtype=jnp.uint32)
# sps = bm.random.randint(0, 2, size).value < 1
# value = 2.
#
#
# # f = jax.jit(event_sum)
#
#
# # @partial(jax.jit, static_argnums=2)
# # def f(sps, pre2post, size, value):
# #   return event_sum(sps, pre2post, size, value)
#
#
# @jax.jit
# def ours(events):
#   out = jnp.zeros(size)
#   out = _event_sum_prim.bind(events, post_ids.value, pre_ids.value, jnp.zeros(1), out)
#   # print(type(out), out)
#   # print(type(value), value)
#   return out
#
#
# # @jax.jity
# # def yours(events):
# #   out = jnp.zeros(size)
# #   out = out.at[post_ids.value].add(events[pre_ids.value])
# #   return out
#
#
# a = ours(sps)
# b = np.asarray(a)
# # print(b)
# print(b.size)
#
# # a = yours(sps)
# # b = np.asarray(a)
# # # print(b)
# # print(b.size)
#
# sps = bm.random.randint(0, 2, size).value < 1
#
# t0 = time.time()
# ours(sps)
# print(time.time() - t0)
#
#
# # t0 = time.time()
# # yours(sps)
# # print(time.time() - t0)
#
# # %timeit f(sps, (indices, indptr), size, value)
# # print(timeit.timeit('ours(sps, value)', globals=globals()))
# # print(timeit.timeit('yours(sps, value)', globals=globals()))
