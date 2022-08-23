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
    class MaxPoolNet(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(MaxPoolNet, self).__init__()
        self.maxpool = bp.dyn.layers.MaxPool((2, 2))

      def update(self, sha, x):
        x = self.maxpool(sha, x)
        return x

    x = jnp.arange(9).reshape((1, 3, 3, 1)).astype(jnp.float32)
    shared = {'fit': False}
    net = MaxPoolNet()
    y = net(shared, x)
    print("out shape: ", y.shape)
    expected_y = jnp.array([
      [4., 5.],
      [7., 8.],
    ]).reshape((1, 2, 2, 1))
    np.testing.assert_allclose(y, expected_y)

  def test_minpool(self):
    class MinPoolNet(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(MinPoolNet, self).__init__()
        self.maxpool = bp.dyn.layers.MinPool((2, 2))

      def update(self, sha, x):
        x = self.maxpool(sha, x)
        return x

    x = jnp.arange(9).reshape((1, 3, 3, 1)).astype(jnp.float32)
    shared = {'fit': False}
    net = MinPoolNet()
    y = net(shared, x)
    print("out shape: ", y.shape)
    expected_y = jnp.array([
      [0., 1.],
      [3., 4.],
    ]).reshape((1, 2, 2, 1))
    np.testing.assert_allclose(y, expected_y)

  def test_avgpool(self):
    class AvgPoolNet(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(AvgPoolNet, self).__init__()
        self.maxpool = bp.dyn.layers.AvgPool((2, 2))

      def update(self, sha, x):
        x = self.maxpool(sha, x)
        return x

    x = jnp.full((1, 3, 3, 1), 2.)
    shared = {'fit': False}
    net = AvgPoolNet()
    y = net(shared, x)
    print("out shape: ", y.shape)
    np.testing.assert_allclose(y, np.full((1, 2, 2, 1), 2.))