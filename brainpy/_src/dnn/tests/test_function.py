# -*- coding: utf-8 -*-

from unittest import TestCase

import jax.numpy as jnp
import brainpy.math as bm
from absl.testing import absltest
import brainpy as bp


class TestFunction(bp.testing.UnitTestCase):

    def test_flatten_batching_mode(self):
        layer = bp.dnn.Flatten(mode=bm.BatchingMode())
        input = bm.random.randn(20, 10, 10, 6)

        output = layer.update(input)

        expected_shape = (20, 600)
        self.assertEqual(output.shape, expected_shape)

    def test_flatten_non_batching_mode(self):
        layer = bp.dnn.Flatten(mode=bm.NonBatchingMode())
        input = bm.random.randn(10, 10, 6)

        output = layer.update(input)

        expected_shape = (600,)
        self.assertEqual(output.shape, expected_shape)


if __name__ == '__main__':
    absltest.main()
