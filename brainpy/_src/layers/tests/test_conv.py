# -*- coding: utf-8 -*-

from unittest import TestCase

import jax.numpy as jnp
import brainpy.math as bm

import brainpy as bp


class TestConv(bp.testing.UnitTestCase):
  def test_Conv2D_img(self):
    img = jnp.zeros((2, 200, 198, 4))
    for k in range(4):
      x = 30 + 60 * k
      y = 20 + 60 * k
      img = img.at[0, x:x + 10, y:y + 10, k].set(1.0)
      img = img.at[1, x:x + 20, y:y + 20, k].set(3.0)

    with bp.math.training_environment():
      net = bp.layers.Conv2d(in_channels=4, out_channels=32, kernel_size=(3, 3),
                             strides=(1, 1), padding='SAME', groups=1)
      out = net(img)
      print("out shape: ", out.shape)
      # print("First output channel:")
      # plt.figure(figsize=(10, 10))
      # plt.imshow(np.array(img)[0, :, :, 0])
      # plt.show()

  def test_conv1D(self):
    with bp.math.training_environment():
      model = bp.layers.Conv1d(in_channels=3, out_channels=32, kernel_size=(3,))
      input = bp.math.ones((2, 5, 3))

      out = model(input)
      print("out shape: ", out.shape)
      # print("First output channel:")
      # plt.figure(figsize=(10, 10))
      # plt.imshow(np.array(out)[0, :, :])
      # plt.show()

  def test_conv2D(self):
    with bp.math.training_environment():
      model = bp.layers.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3))

      input = bp.math.ones((2, 5, 5, 3))

      out = model(input)
      print("out shape: ", out.shape)
      # print("First output channel:")
      # plt.figure(figsize=(10, 10))
      # plt.imshow(np.array(out)[0, :, :, 31])
      # plt.show()

  def test_conv3D(self):
    with bp.math.training_environment():
      model = bp.layers.Conv3d(in_channels=3, out_channels=32, kernel_size=(3, 3, 3))
      input = bp.math.ones((2, 5, 5, 5, 3))
      out = model(input)
      print("out shape: ", out.shape)


class TestConvTranspose1d(bp.testing.UnitTestCase):
  def test_conv_transpose(self):
    x = bm.ones((1, 8, 3))
    for use_bias in [True, False]:
      conv_transpose_module = bp.layers.ConvTranspose1d(
        in_channels=3,
        out_channels=4,
        kernel_size=(3,),
        padding='VALID',
        w_initializer=bp.init.OneInit(),
        b_initializer=bp.init.OneInit() if use_bias else None,
        mode=bm.training_mode
      )
      self.assertEqual(conv_transpose_module.w.shape, (3, 3, 4))
      y = conv_transpose_module(x)
      print(y.shape)
      correct_ans = jnp.array([[[4., 4., 4., 4.],
                                [7., 7., 7., 7.],
                                [10., 10., 10., 10.],
                                [10., 10., 10., 10.],
                                [10., 10., 10., 10.],
                                [10., 10., 10., 10.],
                                [10., 10., 10., 10.],
                                [10., 10., 10., 10.],
                                [7., 7., 7., 7.],
                                [4., 4., 4., 4.]]])
      if not use_bias:
        correct_ans -= 1.
      self.assertTrue(bm.allclose(y, correct_ans))

  def test_single_input_masked_conv_transpose(self):
    x = jnp.ones((1, 8, 3))
    m = jnp.tril(jnp.ones((3, 3, 4)))
    conv_transpose_module = bp.layers.ConvTranspose1d(
      in_channels=3,
      out_channels=4,
      kernel_size=(3,),
      padding='VALID',
      mask=m,
      w_initializer=bp.init.OneInit(),
      b_initializer=bp.init.OneInit(),
      mode=bm.batching_mode
    )
    self.assertEqual(conv_transpose_module.w.shape, (3, 3, 4))
    y = conv_transpose_module(x)
    print(y.shape)
    correct_ans = jnp.array([[[4., 3., 2., 1.],
                              [7., 5., 3., 1.],
                              [10., 7., 4., 1.],
                              [10., 7., 4., 1.],
                              [10., 7., 4., 1.],
                              [10., 7., 4., 1.],
                              [10., 7., 4., 1.],
                              [10., 7., 4., 1.],
                              [7., 5., 3., 1.],
                              [4., 3., 2., 1.]]])
    self.assertTrue(bm.allclose(y, correct_ans))

  def test_computation_padding_same(self):
    data = jnp.ones([1, 3, 1])
    for use_bias in [True, False]:
      net = bp.layers.ConvTranspose1d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        stride=1,
        padding="SAME",
        w_initializer=bp.init.OneInit(),
        b_initializer=bp.init.OneInit() if use_bias else None,
        mode=bm.batching_mode
      )
      out = net(data)
      self.assertEqual(out.shape, (1, 3, 1))
      out = jnp.squeeze(out, axis=(0, 2))
      expected_out = bm.as_jax([2, 3, 2])
      if use_bias:
        expected_out += 1
      self.assertTrue(bm.allclose(out, expected_out, rtol=1e-5))



