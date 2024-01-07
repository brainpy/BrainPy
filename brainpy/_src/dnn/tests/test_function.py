# -*- coding: utf-8 -*-

from unittest import TestCase

import jax.numpy as jnp
import brainpy.math as bm
from absl.testing import absltest
from absl.testing import parameterized
import brainpy as bp


class TestFunction(parameterized.TestCase):

  def test_flatten_batching_mode(self):
    bm.random.seed()
    layer = bp.dnn.Flatten(mode=bm.BatchingMode())
    input = bm.random.randn(20, 10, 10, 6)

    output = layer.update(input)

    expected_shape = (20, 600)
    self.assertEqual(output.shape, expected_shape)
    bm.clear_buffer_memory()

  def test_flatten_non_batching_mode(self):
    bm.random.seed()
    layer = bp.dnn.Flatten(mode=bm.NonBatchingMode())
    input = bm.random.randn(10, 10, 6)

    output = layer.update(input)

    expected_shape = (600,)
    self.assertEqual(output.shape, expected_shape)
    bm.clear_buffer_memory()

  def test_unflatten(self):
    bm.random.seed()
    layer = bp.dnn.Unflatten(1, (10, 6), mode=bm.NonBatchingMode())
    input = bm.random.randn(5, 60)
    output = layer.update(input)
    expected_shape = (5, 10, 6)
    self.assertEqual(output.shape, expected_shape)
    bm.clear_buffer_memory()


if __name__ == '__main__':
  absltest.main()
