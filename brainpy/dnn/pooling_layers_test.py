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
    def test_AdaptiveMaxPool3d_v1(self, axis):
        bm.random.seed()
        input = bm.random.randn(2, 128, 64, 32, 16)
        net = bp.dnn.AdaptiveMaxPool3d(target_shape=[6, 5, 4], channel_axis=axis)
        output = net(input)


if __name__ == '__main__':
    absltest.main()
