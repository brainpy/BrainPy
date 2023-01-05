# -*- coding: utf-8 -*-

from unittest import TestCase

import jax.numpy as jnp

import brainpy as bp


class TestConv(TestCase):
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
