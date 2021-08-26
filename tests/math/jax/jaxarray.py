# -*- coding: utf-8 -*-

import pytest
import jax.numpy as jnp
from brainpy.math.jax import JaxArray


def test_at1():
  # https://jax.readthedocs.io/en/latest/jax.ops.html#indexed-update-operators
  b = jnp.ones(10)
  a = JaxArray(jnp.ones(10))
  b.at[0].set(10)
  a.at[0].set(10)
  assert (a == b).all()

  b.at[1].add(10)
  a.at[1].add(10)
  assert (a == b).all()

  b.at[0].multiply(3)
  a.at[0].multiply(3)
  assert (a == b).all()

  b.at[2].divide(4)
  a.at[2].divide(4)
  assert (a == b).all()

  b.at[2].power(3)
  a.at[2].power(3)
  assert (a == b).all()

  b.at[4].min(0)
  a.at[4].min(0)
  assert (a == b).all()

  b.at[5].max(10)
  a.at[5].max(10)
  assert (a == b).all()

