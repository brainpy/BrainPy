# -*- coding: utf-8 -*-

from unittest import TestCase

import jax.numpy as jnp
import numpy as np

import brainpy as bp
import brainpy.math as bm


class TestPool(TestCase):
  def test_maxpool(self):
    class MaxPoolNet(bp.DynamicalSystem):
      def __init__(self):
        super(MaxPoolNet, self).__init__()
        self.maxpool = bp.layers.MaxPool((2, 2), 1, channel_axis=-1)

      def update(self, sha, x):
        return self.maxpool(sha, x)

    x = jnp.arange(9).reshape((1, 3, 3, 1)).astype(jnp.float32)
    print(jnp.arange(9).reshape(3, 3))
    print(x)
    print(x.shape)
    shared = {'fit': False}
    with bm.env_training():
      net = MaxPoolNet()
    y = net(shared, x)
    print("out shape: ", y.shape)
    expected_y = jnp.array([[4., 5.],
                            [7., 8.]]).reshape((1, 2, 2, 1))
    np.testing.assert_allclose(y, expected_y)

  def test_maxpool2(self):
    class MaxPoolNet(bp.DynamicalSystem):
      def __init__(self):
        super(MaxPoolNet, self).__init__()
        self.maxpool = bp.layers.MaxPool((2, 2), (2, 2), channel_axis=-1)

      def update(self, sha, x):
        return self.maxpool(sha, x)

    rng = bm.random.RandomState(123)
    x = rng.rand(10, 20, 20, 4)
    with bm.env_training():
      net = MaxPoolNet()
    y = net(None, x)
    print("out shape: ", y.shape)

  def test_minpool(self):
    class MinPoolNet(bp.DynamicalSystem):
      def __init__(self):
        super(MinPoolNet, self).__init__()
        self.maxpool = bp.layers.MinPool((2, 2), 1, channel_axis=-1)

      def update(self, sha, x):
        x = self.maxpool(sha, x)
        return x

    x = jnp.arange(9).reshape((1, 3, 3, 1)).astype(jnp.float32)
    shared = {'fit': False}
    with bm.env_training():
      net = MinPoolNet()
    y = net(shared, x)
    print("out shape: ", y.shape)
    expected_y = jnp.array([
      [0., 1.],
      [3., 4.],
    ]).reshape((1, 2, 2, 1))
    np.testing.assert_allclose(y, expected_y)

  def test_avgpool(self):
    class AvgPoolNet(bp.DynamicalSystem):
      def __init__(self):
        super(AvgPoolNet, self).__init__()
        self.maxpool = bp.layers.AvgPool((2, 2), 1, channel_axis=-1)

      def update(self, sha, x):
        x = self.maxpool(sha, x)
        return x

    x = jnp.full((1, 3, 3, 1), 2.)
    shared = {'fit': False}
    with bm.env_training():
      net = AvgPoolNet()
    y = net(shared, x)
    print("out shape: ", y.shape)
    np.testing.assert_allclose(y, np.full((1, 2, 2, 1), 2.))
