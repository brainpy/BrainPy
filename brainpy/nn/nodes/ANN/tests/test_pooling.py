# -*- coding: utf-8 -*-
import random

import pytest
from unittest import TestCase
import brainpy as bp
import jax.numpy as jnp
import jax
import numpy as np


class TestPool(TestCase):
  def test_maxpool(self):
    i = bp.nn.Input((3, 3, 1))
    p = bp.nn.MaxPool((2, 2))
    model = i >> p
    model.initialize(num_batch=1)

    x = jnp.arange(9).reshape((1, 3, 3, 1)).astype(jnp.float32)

    y = model(x)
    print("out shape: ", y.shape)
    expected_y = jnp.array([
      [4., 5.],
      [7., 8.],
    ]).reshape((1, 2, 2, 1))
    np.testing.assert_allclose(y, expected_y)

  def test_minpool(self):
    i = bp.nn.Input((3, 3, 1))
    p = bp.nn.MinPool((2, 2))
    model = i >> p
    model.initialize(num_batch=1)

    x = jnp.arange(9).reshape((1, 3, 3, 1)).astype(jnp.float32)

    y = model(x)
    print("out shape: ", y.shape)
    expected_y = jnp.array([
      [0., 1.],
      [3., 4.],
    ]).reshape((1, 2, 2, 1))
    np.testing.assert_allclose(y, expected_y)

  def test_avgpool(self):
    i = bp.nn.Input((3, 3, 1))
    p = bp.nn.AvgPool((2, 2))
    model = i >> p
    model.initialize(num_batch=1)

    x = jnp.full((1, 3, 3, 1), 2.)
    y = model(x)
    print("out shape: ", y.shape)
    np.testing.assert_allclose(y, np.full((1, 2, 2, 1), 2.))


