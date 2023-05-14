# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm
from brainpy._src.layers import pooling


class TestPool(parameterized.TestCase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.rng = bm.random.default_rng(12345)

  def test_maxpool(self):
    x = jnp.arange(9).reshape((1, 3, 3, 1)).astype(jnp.float32)
    print(jnp.arange(9).reshape(3, 3))
    print(x)
    print(x.shape)
    shared = {'fit': False}
    with bm.training_environment():
      net = bp.layers.MaxPool((2, 2), 1, channel_axis=-1)
    y = net(shared, x)
    print("out shape: ", y.shape)
    expected_y = jnp.array([[4., 5.],
                            [7., 8.]]).reshape((1, 2, 2, 1))
    np.testing.assert_allclose(y, expected_y)

  def test_maxpool2(self):
    x = self.rng.rand(10, 20, 20, 4)
    with bm.training_environment():
      net = bp.layers.MaxPool((2, 2), (2, 2), channel_axis=-1)
    y = net(x)
    print("out shape: ", y.shape)

  def test_minpool(self):
    x = jnp.arange(9).reshape((1, 3, 3, 1)).astype(jnp.float32)
    shared = {'fit': False}
    with bm.training_environment():
      net = bp.layers.MinPool((2, 2), 1, channel_axis=-1)
    y = net(shared, x)
    print("out shape: ", y.shape)
    expected_y = jnp.array([
      [0., 1.],
      [3., 4.],
    ]).reshape((1, 2, 2, 1))
    np.testing.assert_allclose(y, expected_y)

  def test_avgpool(self):
    x = jnp.full((1, 3, 3, 1), 2.)
    with bm.training_environment():
      net = bp.layers.AvgPool((2, 2), 1, channel_axis=-1)
    y = net(x)
    print("out shape: ", y.shape)
    np.testing.assert_allclose(y, np.full((1, 2, 2, 1), 2.))

  def test_MaxPool2d_v1(self):
    arr = self.rng.rand(16, 32, 32, 8)

    out = pooling.MaxPool2d(2, 2, channel_axis=-1)(arr)
    self.assertTrue(out.shape == (16, 16, 16, 8))

    out = pooling.MaxPool2d(2, 2, channel_axis=None)(arr)
    self.assertTrue(out.shape == (16, 32, 16, 4))

    out = pooling.MaxPool2d(2, 2, channel_axis=None, padding=1)(arr)
    self.assertTrue(out.shape == (16, 32, 17, 5))

    out = pooling.MaxPool2d(2, 2, channel_axis=None, padding=(2, 1))(arr)
    self.assertTrue(out.shape == (16, 32, 18, 5))

    out = pooling.MaxPool2d(2, 2, channel_axis=-1, padding=(1, 1))(arr)
    self.assertTrue(out.shape == (16, 17, 17, 8))

    out = pooling.MaxPool2d(2, 2, channel_axis=2, padding=(1, 1))(arr)
    self.assertTrue(out.shape == (16, 17, 32, 5))

  def test_AvgPool2d_v1(self):
    arr = self.rng.rand(16, 32, 32, 8)

    out = pooling.AvgPool2d(2, 2, channel_axis=-1)(arr)
    self.assertTrue(out.shape == (16, 16, 16, 8))

    out = pooling.AvgPool2d(2, 2, channel_axis=None)(arr)
    self.assertTrue(out.shape == (16, 32, 16, 4))

    out = pooling.AvgPool2d(2, 2, channel_axis=None, padding=1)(arr)
    self.assertTrue(out.shape == (16, 32, 17, 5))

    out = pooling.AvgPool2d(2, 2, channel_axis=None, padding=(2, 1))(arr)
    self.assertTrue(out.shape == (16, 32, 18, 5))

    out = pooling.AvgPool2d(2, 2, channel_axis=-1, padding=(1, 1))(arr)
    self.assertTrue(out.shape == (16, 17, 17, 8))

    out = pooling.AvgPool2d(2, 2, channel_axis=2, padding=(1, 1))(arr)
    self.assertTrue(out.shape == (16, 17, 32, 5))

  @parameterized.named_parameters(
    dict(testcase_name=f'target_size={target_size}',
         target_size=target_size)
    for target_size in [10, 9, 8, 7, 6]
  )
  def test_adaptive_pool1d(self, target_size):
    arr = self.rng.rand(100)
    op = jax.numpy.mean

    out = pooling._adaptive_pool1d(arr, target_size, op)
    print(out.shape)
    self.assertTrue(out.shape == (target_size,))

    out = pooling._adaptive_pool1d(arr, target_size, op)
    print(out.shape)
    self.assertTrue(out.shape == (target_size,))

  def test_AdaptiveAvgPool2d_v1(self):
    input = self.rng.randn(64, 8, 9)

    output = pooling.AdaptiveAvgPool2d((5, 7), channel_axis=0)(input)
    self.assertTrue(output.shape == (64, 5, 7))

    output = pooling.AdaptiveAvgPool2d((2, 3), channel_axis=0)(input)
    self.assertTrue(output.shape == (64, 2, 3))

    output = pooling.AdaptiveAvgPool2d((2, 3), channel_axis=-1)(input)
    self.assertTrue(output.shape == (2, 3, 9))

    output = pooling.AdaptiveAvgPool2d((2, 3), channel_axis=1)(input)
    self.assertTrue(output.shape == (2, 8, 3))

    output = pooling.AdaptiveAvgPool2d((2, 3), channel_axis=None)(input)
    self.assertTrue(output.shape == (64, 2, 3))

  def test_AdaptiveAvgPool2d_v2(self):
    input = self.rng.randn(128, 64, 32, 16)

    output = pooling.AdaptiveAvgPool2d((5, 7), channel_axis=0)(input)
    self.assertTrue(output.shape == (128, 64, 5, 7))

    output = pooling.AdaptiveAvgPool2d((2, 3), channel_axis=0)(input)
    self.assertTrue(output.shape == (128, 64, 2, 3))

    output = pooling.AdaptiveAvgPool2d((2, 3), channel_axis=-1)(input)
    self.assertTrue(output.shape == (128, 2, 3, 16))

    output = pooling.AdaptiveAvgPool2d((2, 3), channel_axis=1)(input)
    self.assertTrue(output.shape == (128, 64, 2, 3))
