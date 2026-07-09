# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm


class TestPool(parameterized.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_maxpool(self):
        bm.random.seed()
        x = jnp.arange(9).reshape((1, 3, 3, 1)).astype(jnp.float32)
        print(jnp.arange(9).reshape(3, 3))
        print(x)
        print(x.shape)
        shared = {'fit': False}
        with bm.training_environment():
            net = bp.dnn.MaxPool((2, 2), 1, channel_axis=-1)
        y = net(shared, x)
        print("out shape: ", y.shape)
        expected_y = jnp.array([[4., 5.],
                                [7., 8.]]).reshape((1, 2, 2, 1))
        np.testing.assert_allclose(y, expected_y)

    def test_maxpool2(self):
        bm.random.seed()
        x = bm.random.rand(10, 20, 20, 4)
        with bm.training_environment():
            net = bp.dnn.MaxPool((2, 2), (2, 2), channel_axis=-1)
        y = net(x)
        print("out shape: ", y.shape)

    def test_minpool(self):
        bm.random.seed()
        x = jnp.arange(9).reshape((1, 3, 3, 1)).astype(jnp.float32)
        shared = {'fit': False}
        with bm.training_environment():
            net = bp.dnn.MinPool((2, 2), 1, channel_axis=-1)
        y = net(shared, x)
        print("out shape: ", y.shape)
        expected_y = jnp.array([
            [0., 1.],
            [3., 4.],
        ]).reshape((1, 2, 2, 1))
        np.testing.assert_allclose(y, expected_y)

    def test_avgpool(self):
        bm.random.seed()
        x = jnp.full((1, 3, 3, 1), 2.)
        with bm.training_environment():
            net = bp.dnn.AvgPool((2, 2), 1, channel_axis=-1)
        y = net(x)
        print("out shape: ", y.shape)
        np.testing.assert_allclose(y, np.full((1, 2, 2, 1), 2.))

    def test_MaxPool2d_v1(self):
        bm.random.seed()
        arr = bm.random.rand(16, 32, 32, 8)

        out = bp.dnn.MaxPool2d(2, 2, channel_axis=-1)(arr)
        self.assertTrue(out.shape == (16, 16, 16, 8))

        out = bp.dnn.MaxPool2d(2, 2, channel_axis=None)(arr)
        self.assertTrue(out.shape == (16, 32, 16, 4))

        out = bp.dnn.MaxPool2d(2, 2, channel_axis=None, padding=1)(arr)
        self.assertTrue(out.shape == (16, 32, 17, 5))

        out = bp.dnn.MaxPool2d(2, 2, channel_axis=None, padding=(2, 1))(arr)
        self.assertTrue(out.shape == (16, 32, 18, 5))

        out = bp.dnn.MaxPool2d(2, 2, channel_axis=-1, padding=(1, 1))(arr)
        self.assertTrue(out.shape == (16, 17, 17, 8))

        out = bp.dnn.MaxPool2d(2, 2, channel_axis=2, padding=(1, 1))(arr)
        self.assertTrue(out.shape == (16, 17, 32, 5))

    def test_AvgPool2d_v1(self):
        bm.random.seed()
        arr = bm.random.rand(16, 32, 32, 8)

        out = bp.dnn.AvgPool2d(2, 2, channel_axis=-1)(arr)
        self.assertTrue(out.shape == (16, 16, 16, 8))

        out = bp.dnn.AvgPool2d(2, 2, channel_axis=None)(arr)
        self.assertTrue(out.shape == (16, 32, 16, 4))

        out = bp.dnn.AvgPool2d(2, 2, channel_axis=None, padding=1)(arr)
        self.assertTrue(out.shape == (16, 32, 17, 5))

        out = bp.dnn.AvgPool2d(2, 2, channel_axis=None, padding=(2, 1))(arr)
        self.assertTrue(out.shape == (16, 32, 18, 5))

        out = bp.dnn.AvgPool2d(2, 2, channel_axis=-1, padding=(1, 1))(arr)
        self.assertTrue(out.shape == (16, 17, 17, 8))

        out = bp.dnn.AvgPool2d(2, 2, channel_axis=2, padding=(1, 1))(arr)
        self.assertTrue(out.shape == (16, 17, 32, 5))

    @parameterized.named_parameters(
        dict(testcase_name=f'target_size={target_size}',
             target_size=target_size)
        for target_size in [10, 9, 8, 7, 6]
    )
    def test_adaptive_pool1d(self, target_size):
        bm.random.seed()
        from brainpy.dnn.pooling import _adaptive_pool1d

        arr = bm.random.rand(100)
        op = jax.numpy.mean

        out = _adaptive_pool1d(arr, target_size, op)
        print(out.shape)
        self.assertTrue(out.shape == (target_size,))

        out = _adaptive_pool1d(arr, target_size, op)
        print(out.shape)
        self.assertTrue(out.shape == (target_size,))

    def test_AdaptiveAvgPool2d_v1(self):
        bm.random.seed()
        input = bm.random.randn(64, 8, 9)

        output = bp.dnn.AdaptiveAvgPool2d((5, 7), channel_axis=0)(input)
        self.assertTrue(output.shape == (64, 5, 7))

        output = bp.dnn.AdaptiveAvgPool2d((2, 3), channel_axis=0)(input)
        self.assertTrue(output.shape == (64, 2, 3))

        output = bp.dnn.AdaptiveAvgPool2d((2, 3), channel_axis=-1)(input)
        self.assertTrue(output.shape == (2, 3, 9))

        output = bp.dnn.AdaptiveAvgPool2d((2, 3), channel_axis=1)(input)
        self.assertTrue(output.shape == (2, 8, 3))

        output = bp.dnn.AdaptiveAvgPool2d((2, 3), channel_axis=None)(input)
        self.assertTrue(output.shape == (64, 2, 3))

    def test_AdaptiveAvgPool2d_v2(self):
        bm.random.seed()
        input = bm.random.randn(128, 64, 32, 16)

        output = bp.dnn.AdaptiveAvgPool2d((5, 7), channel_axis=0)(input)
        self.assertTrue(output.shape == (128, 64, 5, 7))

        output = bp.dnn.AdaptiveAvgPool2d((2, 3), channel_axis=0)(input)
        self.assertTrue(output.shape == (128, 64, 2, 3))

        output = bp.dnn.AdaptiveAvgPool2d((2, 3), channel_axis=-1)(input)
        self.assertTrue(output.shape == (128, 2, 3, 16))

        output = bp.dnn.AdaptiveAvgPool2d((2, 3), channel_axis=1)(input)
        self.assertTrue(output.shape == (128, 64, 2, 3))
        print()

    def test_AdaptiveAvgPool3d_v1(self):
        bm.random.seed()
        input = bm.random.randn(10, 128, 64, 32)
        net = bp.dnn.AdaptiveAvgPool3d(target_shape=[6, 5, 3], channel_axis=0, mode=bm.nonbatching_mode)
        output = net(input)
        self.assertTrue(output.shape == (10, 6, 5, 3))

    def test_AdaptiveAvgPool3d_v2(self):
        bm.random.seed()
        input = bm.random.randn(10, 20, 128, 64, 32)
        net = bp.dnn.AdaptiveAvgPool3d(target_shape=[6, 5, 3], mode=bm.batching_mode)
        output = net(input)
        self.assertTrue(output.shape == (10, 6, 5, 3, 32))

    @parameterized.product(
        axis=(-1, 0, 1)
    )
    def test_AdaptiveMaxPool1d_v1(self, axis):
        bm.random.seed()
        input = bm.random.randn(32, 16)
        net = bp.dnn.AdaptiveMaxPool1d(target_shape=4, channel_axis=axis)
        output = net(input)

    @parameterized.product(
        axis=(-1, 0, 1, 2)
    )
    def test_AdaptiveMaxPool1d_v2(self, axis):
        bm.random.seed()
        input = bm.random.randn(2, 32, 16)
        net = bp.dnn.AdaptiveMaxPool1d(target_shape=4, channel_axis=axis)
        output = net(input)

    @parameterized.product(
        axis=(-1, 0, 1, 2)
    )
    def test_AdaptiveMaxPool2d_v1(self, axis):
        bm.random.seed()
        input = bm.random.randn(32, 16, 12)
        net = bp.dnn.AdaptiveAvgPool2d(target_shape=[5, 4], channel_axis=axis)
        output = net(input)

    @parameterized.product(
        axis=(-1, 0, 1, 2, 3)
    )
    def test_AdaptiveMaxPool2d_v2(self, axis):
        bm.random.seed()
        input = bm.random.randn(2, 32, 16, 12)
        net = bp.dnn.AdaptiveAvgPool2d(target_shape=[5, 4], channel_axis=axis)
        # output = net(input)

    @parameterized.product(
        axis=(-1, 0, 1, 2, 3)
    )
    def test_AdaptiveMaxPool3d_v1(self, axis):
        bm.random.seed()
        input = bm.random.randn(2, 128, 64, 32)
        net = bp.dnn.AdaptiveMaxPool3d(target_shape=[6, 5, 4], channel_axis=axis)
        output = net(input)
        print()

    @parameterized.product(
        axis=(-1, 0, 1, 2, 3, 4)
    )
    def test_AdaptiveMaxPool3d_v2(self, axis):
        bm.random.seed()
        input = bm.random.randn(2, 128, 64, 32, 16)
        net = bp.dnn.AdaptiveMaxPool3d(target_shape=[6, 5, 4], channel_axis=axis)
        output = net(input)


class TestPoolingChannelAxis(parameterized.TestCase):
    """Regression for P12-M2: the leftmost negative ``channel_axis`` (== -x_dim)
    was wrongly rejected because the bound check used ``abs(channel_axis)``."""

    def test_maxpool2d_leftmost_negative_channel_axis(self):
        bm.random.seed()
        # channels-first (C, H, W); channel axis is axis 0 == -3.
        x = bm.random.randn(6, 8, 8)
        net = bp.dnn.MaxPool2d(2, channel_axis=-3)
        out = net(x)
        self.assertEqual(out.shape, (6, 4, 4))
        # Must equal the equivalent positive channel_axis result.
        net_pos = bp.dnn.MaxPool2d(2, channel_axis=0)
        self.assertEqual(out.shape, net_pos(x).shape)

    def test_adaptiveavgpool2d_leftmost_negative_channel_axis(self):
        bm.random.seed()
        x = bm.random.randn(6, 8, 8)  # (C, H, W)
        net = bp.dnn.AdaptiveAvgPool2d((4, 4), channel_axis=-3)
        out = net(x)
        self.assertEqual(out.shape, (6, 4, 4))

    def test_pool_leftmost_negative_channel_axis(self):
        bm.random.seed()
        # ``Pool`` family (MaxPool) with an integer kernel and channel_axis=-3.
        x = bm.random.randn(6, 8, 8)
        net = bp.dnn.MaxPool((2, 2), 2, channel_axis=-3)
        out = net(x)
        self.assertEqual(out.shape, (6, 4, 4))


def _adaptive_pool1d_reference(x, target_size, op):
    """Reference PyTorch-style adaptive pooling of a 1-D array (numpy)."""
    x = np.asarray(x)
    size = x.shape[0]
    out = []
    for i in range(target_size):
        start = (i * size) // target_size
        end = -((-((i + 1) * size)) // target_size)  # ceil((i + 1) * size / target_size)
        out.append(op(x[start:end]))
    return np.array(out)


class TestAdaptivePool1d(parameterized.TestCase):
    """Regression coverage for ``_adaptive_pool1d``.

    Guards the fix for the ``ZeroDivisionError: integer modulo by zero`` that arose
    when ``target_size > size`` (a spatial dimension smaller than its target), which
    made the old block-reshape implementation build ``reshape(-1, 0)``.
    """

    @parameterized.product(
        size_target=((100, 6), (100, 7), (100, 10), (32, 4), (5, 5),
                     (2, 6), (1, 4), (3, 8)),
        op=('mean', 'max'),
    )
    def test_matches_pytorch_formula(self, size_target, op):
        from brainpy.dnn.pooling import _adaptive_pool1d
        size, target = size_target
        jop, nop = (jnp.mean, np.mean) if op == 'mean' else (jnp.max, np.max)
        x = np.arange(size, dtype=np.float32) * 0.5 - 3.0
        got = np.asarray(_adaptive_pool1d(bm.as_jax(x), target, jop))
        expected = _adaptive_pool1d_reference(x, target, nop)
        self.assertEqual(got.shape, (target,))
        np.testing.assert_allclose(got, expected, atol=1e-5)

    def test_upsampling_repeats_elements(self):
        # target_size (6) > size (2): the previously-crashing case. PyTorch adaptive
        # max pooling repeats each element across its bins.
        from brainpy.dnn.pooling import _adaptive_pool1d
        x = jnp.asarray([10.0, 20.0])
        out = np.asarray(_adaptive_pool1d(x, 6, jnp.max))
        np.testing.assert_array_equal(out, [10., 10., 10., 20., 20., 20.])

    def test_rejects_nonpositive_target(self):
        from brainpy.dnn.pooling import _adaptive_pool1d
        with self.assertRaises(ValueError):
            _adaptive_pool1d(jnp.arange(4.0), 0, jnp.mean)
        with self.assertRaises(ValueError):
            _adaptive_pool1d(jnp.arange(4.0), -2, jnp.mean)

    @parameterized.product(axis=(-1, 0, 1, 2, 3))
    def test_adaptivemaxpool3d_spatial_dim_smaller_than_target(self, axis):
        # A spatial dim of size 2 is pooled to target 6 for every channel_axis that
        # does not consume it; this raised ZeroDivisionError before the fix.
        bm.random.seed(123)
        inp = bm.random.randn(2, 128, 64, 32)
        net = bp.dnn.AdaptiveMaxPool3d(target_shape=[6, 5, 4], channel_axis=axis)
        out = net(inp)
        channel_size = inp.shape[axis]
        self.assertEqual(sorted(out.shape), sorted([channel_size, 6, 5, 4]))
        self.assertTrue(bool(jnp.all(jnp.isfinite(out))))


if __name__ == '__main__':
    absltest.main()
