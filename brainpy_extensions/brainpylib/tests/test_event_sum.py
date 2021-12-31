# -*- coding: utf-8 -*-

import numpy as np
import brainpy.math as bm
from brainpylib import event_sum
import jax.numpy as jnp
import pytest


def test1():
  bm.random.seed(12345)
  size = 200
  indices = jnp.arange(size, dtype=jnp.uint32)
  idnptr = jnp.arange(size + 1, dtype=jnp.uint32)
  sps = bm.random.randint(0, 2, size).value < 1
  a = event_sum(sps,
                (indices, idnptr),
                size,
                1.)
  b = np.asarray(a)
  print(a)
