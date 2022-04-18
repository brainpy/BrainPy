# -*- coding: utf-8 -*-
import random

import pytest
from unittest import TestCase
import brainpy as bp
import jax.numpy as jnp
import numpy as np

class TestConv(TestCase):
  def test_Conv2D_img(self):
    i = bp.nn.Input((200, 198, 4))
    b = bp.nn.Conv2D(32, (3, 3), strides=(1, 1), padding='VALID', groups=2)
    model = i >> b
    model.initialize(num_batch=2)

    img = jnp.zeros((2, 200, 198, 4))
    for k in range(4):
      x = 30 + 60 * k
      y = 20 + 60 * k
      img = img.at[0, x:x + 10, y:y + 10, k].set(1.0)
      img = img.at[1, x:x + 20, y:y + 20, k].set(3.0)

    out = model(img)
    print("out shape: ", out.shape)
    # print("First output channel:")
    # plt.figure(figsize=(10, 10))
    # plt.imshow(np.array(out)[0, :, :, 0])
    # plt.show()

  def test_conv2D_fb(self):
    i = bp.nn.Input((5, 5, 3))
    b = bp.nn.Conv2D(32, (3, 3))
    c = bp.nn.Conv2D(64, (3, 3))
    model = (i >> b >> c) & (b << c)
    model.initialize(num_batch=2)

    input = bp.math.ones((2, 5, 5, 3))

    out = model(input)
    print("out shape: ", out.shape)

  def test_conv1D(self):
    i = bp.nn.Input((5, 3))
    b = bp.nn.Conv1D(32, (3,))
    model = i >> b
    model.initialize(num_batch=2)

    input = bp.math.ones((2, 5, 3))

    out = model(input)
    print("out shape: ", out.shape)
    # print("First output channel:")
    # plt.figure(figsize=(10, 10))
    # plt.imshow(np.array(out)[0, :, :])
    # plt.show()

  def test_conv2D(self):
    i = bp.nn.Input((5, 5, 3))
    b = bp.nn.Conv2D(32, (3, 3))
    model = i >> b
    model.initialize(num_batch=2)

    input = bp.math.ones((2, 5, 5, 3))

    out = model(input)
    print("out shape: ", out.shape)
    # print("First output channel:")
    # plt.figure(figsize=(10, 10))
    # plt.imshow(np.array(out)[0, :, :, 31])
    # plt.show()

  def test_conv3D(self):
    i = bp.nn.Input((5, 5, 5, 3))
    b = bp.nn.Conv3D(32, (3, 3, 3))
    model = i >> b
    model.initialize(num_batch=2)

    input = bp.math.ones((2, 5, 5, 5, 3))

    out = model(input)
    print("out shape: ", out.shape)
