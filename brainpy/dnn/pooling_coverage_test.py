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
"""Coverage tests for ``brainpy/dnn/pooling.py``.

Target: validation/error branches and the 1d/3d variants that
``pooling_layers_test.py`` does not reach. Exercised here:

* ``Pool.__init__`` padding validation (invalid string, non-tuple sequence,
  wrong-length tuple).
* ``Pool._infer_shape`` branches: invalid channel axis, oversized window,
  batching-mode prepend, full-dimension window, int-window paths, the
  "channel_axis should be provided" error and the "provide more elements" error.
* ``AvgPool.update`` SAME-padding window-count branch.
* ``_MaxPoolNd.__init__`` validation: kernel_size/stride length + type errors,
  and the full set of padding parsing branches.
* ``_MaxPoolNd.update`` / ``_AvgPoolNd.update`` "too few dimensions" errors and
  ``_infer_shape`` invalid-channel-axis error.
* The ``MaxPool1d``/``MaxPool3d``/``AvgPool1d``/``AvgPool3d`` constructors and
  forward passes.
* ``AdaptivePool`` target-shape validation, invalid-channel-axis and
  too-few-dimensions errors, plus ``AdaptiveAvgPool1d`` and
  ``AdaptiveMaxPool2d`` forward passes.

All inputs are tiny (spatial dims <= 9, channels 1-2).
"""

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm


class TestPoolInit(parameterized.TestCase):
    def test_invalid_str_padding(self):
        with self.assertRaises(ValueError):
            bp.dnn.MaxPool((2, 2), 1, padding='BAD', channel_axis=-1)

    def test_sequence_padding_not_tuples(self):
        with self.assertRaises(AssertionError):
            bp.dnn.MaxPool((2, 2), 1, padding=[1, 2], channel_axis=-1)

    def test_sequence_padding_bad_pair_length(self):
        with self.assertRaises(AssertionError):
            bp.dnn.MaxPool((2, 2), 1, padding=[(1, 2, 3), (1, 2, 3)], channel_axis=-1)


class TestPoolInferShape(parameterized.TestCase):
    def test_invalid_channel_axis(self):
        net = bp.dnn.MaxPool((2, 2), 1, channel_axis=5, mode=bm.nonbatching_mode)
        with self.assertRaises(ValueError):
            net(jnp.ones((1, 3, 3, 1)))

    def test_window_bigger_than_input(self):
        net = bp.dnn.MaxPool((2, 2, 2, 2, 2), 1, channel_axis=-1, mode=bm.nonbatching_mode)
        with self.assertRaises(ValueError):
            net(jnp.ones((1, 3, 3, 1)))

    def test_full_dimension_window(self):
        # len(size) == x_dim returns size directly.
        net = bp.dnn.MaxPool((1, 2, 2, 1), 1, channel_axis=-1, mode=bm.nonbatching_mode)
        y = net(jnp.ones((1, 3, 3, 1)))
        self.assertEqual(y.shape, (1, 2, 2, 1))

    def test_batching_mode_tuple_window(self):
        # BatchingMode prepends element; channel_axis insertion path.
        net = bp.dnn.MaxPool((2, 2), 1, channel_axis=-1, mode=bm.batching_mode)
        y = net(jnp.ones((1, 3, 3, 1)))
        self.assertEqual(y.shape, (1, 2, 2, 1))

    def test_batching_mode_tuple_window_channel_axis_none_raises(self):
        net = bp.dnn.MaxPool((2, 2), 1, channel_axis=None, mode=bm.batching_mode)
        with self.assertRaises(ValueError):
            net(jnp.ones((1, 3, 3, 1)))

    def test_size_provide_more_elements_raises(self):
        # nonbatching, tuple window len two short of x_dim hits the else ValueError.
        net = bp.dnn.MaxPool((2, 2), 1, channel_axis=-1, mode=bm.nonbatching_mode)
        with self.assertRaises(ValueError):
            net(jnp.ones((1, 3, 3, 1)))

    def test_int_window_nonbatching(self):
        net = bp.dnn.MaxPool(2, 1, channel_axis=-1, mode=bm.nonbatching_mode)
        y = net(jnp.ones((3, 3, 1)))
        self.assertEqual(y.shape, (2, 2, 1))

    def test_int_window_batching(self):
        net = bp.dnn.MaxPool(2, 1, channel_axis=-1, mode=bm.batching_mode)
        y = net(jnp.ones((1, 3, 3, 1)))
        self.assertEqual(y.shape, (1, 2, 2, 1))


class TestAvgPoolUpdate(parameterized.TestCase):
    def test_same_padding_window_counts(self):
        # SAME padding takes the window-count normalisation branch.
        with bm.training_environment():
            net = bp.dnn.AvgPool((2, 2), 1, padding='SAME', channel_axis=-1)
        y = net(jnp.ones((1, 3, 3, 1)))
        self.assertEqual(y.shape, (1, 3, 3, 1))
        # Every window of ones averages to 1.
        np.testing.assert_allclose(np.asarray(y), np.ones((1, 3, 3, 1)), rtol=1e-5)

    def test_valid_padding(self):
        with bm.training_environment():
            net = bp.dnn.AvgPool((2, 2), 1, padding='VALID', channel_axis=-1)
        y = net(jnp.full((1, 3, 3, 1), 2.))
        np.testing.assert_allclose(np.asarray(y), np.full((1, 2, 2, 1), 2.), rtol=1e-5)


class TestMaxPoolNdInit(parameterized.TestCase):
    def test_kernel_size_wrong_length(self):
        with self.assertRaises(ValueError):
            bp.dnn.MaxPool2d((2, 2, 2))

    def test_kernel_size_wrong_type(self):
        with self.assertRaises(TypeError):
            bp.dnn.MaxPool2d(2.5)

    def test_stride_wrong_length(self):
        with self.assertRaises(ValueError):
            bp.dnn.MaxPool2d(2, stride=(1, 1, 1))

    def test_stride_wrong_type(self):
        with self.assertRaises(TypeError):
            bp.dnn.MaxPool2d(2, stride=2.5)

    def test_padding_invalid_str(self):
        with self.assertRaises(ValueError):
            bp.dnn.MaxPool2d(2, padding='BAD')

    def test_padding_int(self):
        self.assertEqual(bp.dnn.MaxPool2d(2, padding=1).padding, [(1, 1), (1, 1)])

    def test_padding_sequence_of_ints(self):
        self.assertEqual(bp.dnn.MaxPool2d(2, padding=[1, 2]).padding, [(1, 1), (2, 2)])

    def test_padding_sequence_of_ints_wrong_length(self):
        with self.assertRaises(ValueError):
            bp.dnn.MaxPool2d(2, padding=[1, 2, 3])

    def test_padding_sequence_not_all_tuples(self):
        with self.assertRaises(ValueError):
            bp.dnn.MaxPool2d(2, padding=[(1, 1), 2])

    def test_padding_tuple_entries_wrong_length(self):
        with self.assertRaises(ValueError):
            bp.dnn.MaxPool2d(2, padding=[(1, 1, 1), (2, 2, 2)])

    def test_padding_sequence_length_one(self):
        self.assertEqual(bp.dnn.MaxPool2d(2, padding=[(1, 2)]).padding, ((1, 2), (1, 2)))

    def test_padding_bad_type(self):
        with self.assertRaises(ValueError):
            bp.dnn.MaxPool2d(2, padding=2.5)


class TestMaxPoolNdUpdate(parameterized.TestCase):
    def test_too_few_dimensions(self):
        net = bp.dnn.MaxPool1d(2, channel_axis=-1)
        with self.assertRaises(ValueError):
            net(jnp.ones((3,)))

    def test_infer_shape_invalid_channel_axis(self):
        net = bp.dnn.MaxPool1d(2, channel_axis=5)
        with self.assertRaises(ValueError):
            net(jnp.ones((4, 3)))

    def test_avgpool_too_few_dimensions(self):
        net = bp.dnn.AvgPool1d(2, channel_axis=-1)
        with self.assertRaises(ValueError):
            net(jnp.ones((3,)))


class TestPoolNdVariants(parameterized.TestCase):
    def test_maxpool1d(self):
        y = bp.dnn.MaxPool1d(2, channel_axis=-1)(jnp.ones((4, 3)))
        self.assertEqual(y.shape, (2, 3))

    def test_maxpool3d(self):
        y = bp.dnn.MaxPool3d(2, channel_axis=-1)(jnp.ones((4, 4, 4, 2)))
        self.assertEqual(y.shape, (2, 2, 2, 2))

    def test_avgpool1d(self):
        y = bp.dnn.AvgPool1d(2, channel_axis=-1)(jnp.ones((4, 3)))
        self.assertEqual(y.shape, (3, 3))

    def test_avgpool3d(self):
        y = bp.dnn.AvgPool3d(2, channel_axis=-1)(jnp.ones((4, 4, 4, 2)))
        self.assertEqual(y.shape, (3, 3, 3, 2))


class TestAdaptivePool(parameterized.TestCase):
    def test_target_shape_invalid(self):
        with self.assertRaises(ValueError):
            bp.dnn.AdaptiveAvgPool2d((2, 3, 4))

    def test_invalid_channel_axis(self):
        net = bp.dnn.AdaptiveAvgPool2d((2, 3), channel_axis=5)
        with self.assertRaises(ValueError):
            net(jnp.ones((4, 5, 6)))

    def test_input_dim_too_small(self):
        # channel_axis=None skips the channel-axis block, hitting the dim check.
        net = bp.dnn.AdaptiveAvgPool2d((2, 3), channel_axis=None)
        with self.assertRaises(ValueError):
            net(jnp.ones((4,)))

    def test_channel_axis_none_forward(self):
        net = bp.dnn.AdaptiveAvgPool2d((2, 3), channel_axis=None)
        y = net(jnp.ones((4, 8, 9)))
        self.assertEqual(y.shape, (4, 2, 3))

    def test_adaptive_avg_pool1d(self):
        y = bp.dnn.AdaptiveAvgPool1d(3, channel_axis=-1)(jnp.ones((8, 2)))
        self.assertEqual(y.shape, (3, 2))

    def test_adaptive_max_pool2d(self):
        y = bp.dnn.AdaptiveMaxPool2d((2, 3), channel_axis=-1)(jnp.ones((8, 9, 2)))
        self.assertEqual(y.shape, (2, 3, 2))


if __name__ == '__main__':
    absltest.main()
