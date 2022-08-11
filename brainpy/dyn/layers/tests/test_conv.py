# -*- coding: utf-8 -*-
import random

import pytest
from unittest import TestCase
import brainpy as bp
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


class TestConv(TestCase):
  def test_Conv2D_img(self):
    class Convnet(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(Convnet, self).__init__()
        self.conv = bp.layers.Conv2D(in_channels=4, out_channels=32, kernel_size=(3, 3),
                                     strides=(1, 1), padding='SAME', groups=1)

      def update(self, shared, x):
        x = self.conv(shared, x)
        return x

    img = jnp.zeros((2, 200, 198, 4))
    for k in range(4):
      x = 30 + 60 * k
      y = 20 + 60 * k
      img = img.at[0, x:x + 10, y:y + 10, k].set(1.0)
      img = img.at[1, x:x + 20, y:y + 20, k].set(3.0)

    net = Convnet()
    out = net(None, img)
    print("out shape: ", out.shape)
    # print("First output channel:")
    # plt.figure(figsize=(10, 10))
    # plt.imshow(np.array(img)[0, :, :, 0])
    # plt.show()

  def test_conv1D(self):
    class Convnet(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(Convnet, self).__init__()
        self.conv = bp.layers.Conv1D(in_channels=3, out_channels=32, kernel_size=(3,))

      def update(self, shared, x):
        x = self.conv(shared, x)
        return x

    model = Convnet()
    input = bp.math.ones((2, 5, 3))

    out = model(None, input)
    print("out shape: ", out.shape)
    # print("First output channel:")
    # plt.figure(figsize=(10, 10))
    # plt.imshow(np.array(out)[0, :, :])
    # plt.show()

  def test_conv2D(self):
    class Convnet(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(Convnet, self).__init__()
        self.conv = bp.layers.Conv2D(in_channels=3, out_channels=32, kernel_size=(3, 3))

      def update(self, shared, x):
        x = self.conv(shared, x)
        return x

    model = Convnet()

    input = bp.math.ones((2, 5, 5, 3))

    out = model(None, input)
    print("out shape: ", out.shape)
    # print("First output channel:")
    # plt.figure(figsize=(10, 10))
    # plt.imshow(np.array(out)[0, :, :, 31])
    # plt.show()

  def test_conv3D(self):
    class Convnet(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(Convnet, self).__init__()
        self.conv = bp.layers.Conv3D(in_channels=3, out_channels=32, kernel_size=(3, 3, 3))

      def update(self, shared, x):
        x = self.conv(shared, x)
        return x

    model = Convnet()

    input = bp.math.ones((2, 5, 5, 5, 3))

    out = model(None, input)
    print("out shape: ", out.shape)
