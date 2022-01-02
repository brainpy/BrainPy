# -*- coding: utf-8 -*-

import numpy as np
import brainpy.math as bm
from brainpylib import atomic_sum
import jax.numpy as jnp
import pytest


def test1():
  bm.random.seed(12345)
  size = 200
  post_ids = jnp.arange(size, dtype=jnp.uint32)
  pre_ids = jnp.arange(size, dtype=jnp.uint32)
  sps = bm.asarray(bm.random.randint(0, 2, size), dtype=bm.float_)
  a = atomic_sum(sps.value, pre_ids, post_ids, size)
  b = np.asarray(a)
  print(a)
