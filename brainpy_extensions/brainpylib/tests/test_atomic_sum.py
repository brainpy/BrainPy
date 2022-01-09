# -*- coding: utf-8 -*-


import brainpy as bp
from brainpylib import atomic_sum
import jax.numpy as jnp
import pytest
import unittest

bp.math.set_platform('gpu')

class TestAtomicSum(unittest.TestCase):
  def test_heter_values1(self):
    bp.math.random.seed(12345)
    size = 200
    post_ids = jnp.arange(size, dtype=jnp.uint32)
    pre_ids = jnp.arange(size, dtype=jnp.uint32)
    sps = bp.math.asarray(bp.math.random.randint(0, 2, size),
                          dtype=bp.math.float_)
    a = atomic_sum(sps.value, post_ids, size, pre_ids)
    print(a)
    self.assertTrue(jnp.array_equal(a, sps.value))

  def test_homo_value1(self):
    size = 200
    value = 2.
    post_ids = jnp.arange(size, dtype=jnp.uint32)
    a = atomic_sum(value, post_ids, size)
    print(a)
    self.assertTrue(jnp.all(a == value))
