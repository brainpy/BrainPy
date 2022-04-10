# -*- coding: utf-8 -*-
import random

import pytest
from unittest import TestCase
import brainpy as bp
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

class TestConv(TestCase):
  def test_Conv2D(self):
    i = bp.nn.Input((200, 198, 3))
    b = bp.nn.Conv2D(3, 32, (3, 3))
    model = i >> b
    model.initialize(num_batch=2)

    img = jnp.zeros((2, 200, 198, 3), dtype=jnp.float32)
    for k in range(3):
      x = 30 + 60 * k
      y = 20 + 60 * k
      img = img.at[0, x:x + 10, y:y + 10, k].set(1.0)
      img = img.at[1, x:x + 20, y:y + 20, k].set(3.0)

    # img = jnp.transpose(img, [0, 3, 1, 2])

    out = model(img)
    print("out shape: ", out.shape)
    print(out[0])
    # print("First output channel:")
    # plt.figure(figsize=(10, 10))
    # plt.imshow(np.array(out)[0, :, :, 0])
    # plt.show()

  def test_conv1D(self):
    i = bp.nn.Input((5, 3))
    b = bp.nn.Conv1D(3, 32, (3,))
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
    b = bp.nn.Conv2D(3, 32, (3, 3))
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
    b = bp.nn.Conv3D(3, 32, (3, 3, 3))
    model = i >> b
    model.initialize(num_batch=2)

    input = bp.math.ones((2, 5, 5, 5, 3))

    out = model(input)
    print("out shape: ", out.shape)
