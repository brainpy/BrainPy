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
"""Coverage tests for ``brainpy/dnn/conv.py``.

Target: lift line coverage of the convolution layers (``Conv{1,2,3}d`` and
``ConvTranspose{1,2,3}d``) plus the shared ``_GeneralConv`` /
``_GeneralConvTranspose`` bases and the ``to_dimension_numbers`` helper.

What is exercised here (and not by ``conv_layers_test.py``):

* ``to_dimension_numbers`` with ``channels_last=False`` and ``transpose=True``.
* All ``padding`` parsing branches: ``int``, ``Tuple[int, int]``, length-1
  sequence of tuples, full sequence of tuples, the length-mismatch ``ValueError``
  and the wrong-type ``ValueError``.
* ``stride`` / ``strides`` deprecation handling, including the
  "cannot provide both" error.
* Non-batching forward path (input ndim == num_spatial_dims + 1), with and
  without bias.
* The optional weight ``mask`` path and its shape-mismatch ``ValueError``.
* ``_check_input_dim`` error branches (wrong ndim, wrong channels) for the base
  class and every subclass.
* ``__repr__`` of both bases.

Tiny tensors are used throughout (batch 1, spatial <= 6, few channels).
"""

import jax.numpy as jnp
import pytest
from absl.testing import absltest
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm
from brainpy.dnn.conv import to_dimension_numbers, _GeneralConv, _GeneralConvTranspose


class TestToDimensionNumbers(parameterized.TestCase):
    def test_channels_first(self):
        # channels_last=False branch (lines 42-43).
        dn = to_dimension_numbers(2, channels_last=False, transpose=False)
        self.assertEqual(tuple(dn.lhs_spec), (0, 1, 2, 3))
        self.assertEqual(tuple(dn.rhs_spec), (3, 2, 0, 1))

    def test_transpose_kernel(self):
        # transpose=True branch (line 45).
        dn = to_dimension_numbers(2, channels_last=True, transpose=True)
        self.assertEqual(tuple(dn.rhs_spec), (2, 3, 0, 1))


class TestGeneralConvBase(parameterized.TestCase):
    """Directly drive the ``_GeneralConv`` base to hit its own branches."""

    def test_base_forward_nonbatching(self):
        bm.random.seed()
        net = _GeneralConv(num_spatial_dims=1, in_channels=2, out_channels=3,
                           kernel_size=3, mode=bm.nonbatching_mode)
        y = net(jnp.ones((5, 2)))
        self.assertEqual(y.shape, (5, 3))
        # __repr__ of the base (lines 198-201).
        self.assertIn('in_channels=2', repr(net))

    def test_base_check_input_dim_bad_ndim(self):
        # base _check_input_dim wrong-ndim branch (lines 165-167).
        net = _GeneralConv(num_spatial_dims=1, in_channels=2, out_channels=3,
                           kernel_size=3, mode=bm.nonbatching_mode)
        with self.assertRaises(ValueError):
            net(jnp.ones((2, 3, 4, 5)))

    def test_base_check_input_dim_bad_channels(self):
        # base _check_input_dim wrong-channel branch (lines 168-170).
        net = _GeneralConv(num_spatial_dims=1, in_channels=2, out_channels=3,
                           kernel_size=3, mode=bm.nonbatching_mode)
        with self.assertRaises(ValueError):
            net(jnp.ones((5, 9)))


class TestConvPaddingParsing(parameterized.TestCase):
    def test_padding_int(self):
        net = bp.layers.Conv2d(2, 3, kernel_size=3, padding=2, mode=bm.nonbatching_mode)
        self.assertEqual(net.padding, ((2, 2), (2, 2)))

    def test_padding_single_pair(self):
        net = bp.layers.Conv2d(2, 3, kernel_size=3, padding=(1, 2), mode=bm.nonbatching_mode)
        self.assertEqual(net.padding, ((1, 2), (1, 2)))

    def test_padding_length_one_sequence(self):
        net = bp.layers.Conv2d(2, 3, kernel_size=3, padding=[(1, 1)], mode=bm.nonbatching_mode)
        self.assertEqual(net.padding, ((1, 1), (1, 1)))

    def test_padding_full_sequence(self):
        net = bp.layers.Conv2d(2, 3, kernel_size=3, padding=[(1, 1), (2, 2)], mode=bm.nonbatching_mode)
        self.assertEqual(net.padding, ((1, 1), (2, 2)))

    def test_padding_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            bp.layers.Conv2d(2, 3, kernel_size=3, padding=[(1, 1), (2, 2), (3, 3)],
                             mode=bm.nonbatching_mode)

    def test_padding_bad_type_raises(self):
        with self.assertRaises(ValueError):
            bp.layers.Conv2d(2, 3, kernel_size=3, padding=3.5, mode=bm.nonbatching_mode)


class TestConvStrideDeprecation(parameterized.TestCase):
    @parameterized.product(klass=['Conv1d', 'Conv2d', 'Conv3d'])
    def test_strides_keyword(self, klass):
        ndim = {'Conv1d': 1, 'Conv2d': 2, 'Conv3d': 3}[klass]
        layer = getattr(bp.layers, klass)
        net = layer(2, 3, kernel_size=3, strides=2, mode=bm.nonbatching_mode)
        self.assertEqual(net.stride, (2,) * ndim)

    @parameterized.product(klass=['Conv1d', 'Conv2d', 'Conv3d'])
    def test_stride_and_strides_both_raises(self, klass):
        layer = getattr(bp.layers, klass)
        with self.assertRaises(ValueError):
            layer(2, 3, kernel_size=3, stride=2, strides=2, mode=bm.nonbatching_mode)


class TestConvForwardPaths(parameterized.TestCase):
    def test_conv1d_nonbatching_with_bias(self):
        bm.random.seed()
        net = bp.layers.Conv1d(2, 3, kernel_size=3, mode=bm.nonbatching_mode)
        y = net(jnp.ones((5, 2)))
        self.assertEqual(y.shape, (5, 3))

    def test_conv1d_nonbatching_no_bias(self):
        bm.random.seed()
        net = bp.layers.Conv1d(2, 3, kernel_size=3, b_initializer=None, mode=bm.nonbatching_mode)
        y = net(jnp.ones((5, 2)))
        self.assertEqual(y.shape, (5, 3))

    def test_conv2d_batched_no_bias(self):
        bm.random.seed()
        net = bp.layers.Conv2d(2, 3, kernel_size=3, b_initializer=None, mode=bm.batching_mode)
        y = net(jnp.ones((1, 5, 5, 2)))
        self.assertEqual(y.shape, (1, 5, 5, 3))

    def test_conv_with_mask(self):
        bm.random.seed()
        mask = jnp.ones((3, 2, 3))
        net = bp.layers.Conv1d(2, 3, kernel_size=3, mask=mask, mode=bm.nonbatching_mode)
        y = net(jnp.ones((5, 2)))
        self.assertEqual(y.shape, (5, 3))

    def test_conv_mask_shape_mismatch_raises(self):
        net = bp.layers.Conv1d(2, 3, kernel_size=3, mask=jnp.ones((9, 9, 9, 9)),
                               mode=bm.nonbatching_mode)
        with self.assertRaises(ValueError):
            net(jnp.ones((5, 2)))


class TestConvCheckInputDim(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(testcase_name='Conv1d', klass='Conv1d', bad=(2, 3, 4, 5), chans=(5, 9)),
        dict(testcase_name='Conv2d', klass='Conv2d', bad=(2, 3), chans=(5, 5, 9)),
        dict(testcase_name='Conv3d', klass='Conv3d', bad=(2, 3), chans=(5, 5, 5, 9)),
    )
    def test_bad_ndim_and_channels(self, klass, bad, chans):
        layer = getattr(bp.layers, klass)
        net = layer(2, 3, kernel_size=3, mode=bm.nonbatching_mode)
        with self.assertRaises(ValueError):
            net(jnp.ones(bad))
        with self.assertRaises(ValueError):
            net(jnp.ones(chans))


class TestConvTransposePaddingParsing(parameterized.TestCase):
    def test_padding_int(self):
        net = bp.layers.ConvTranspose2d(2, 3, kernel_size=3, padding=2, mode=bm.nonbatching_mode)
        self.assertEqual(net.padding, ((2, 2), (2, 2)))

    def test_padding_single_pair(self):
        net = bp.layers.ConvTranspose2d(2, 3, kernel_size=3, padding=(1, 2), mode=bm.nonbatching_mode)
        self.assertEqual(net.padding, ((1, 2), (1, 2)))

    def test_padding_length_one_sequence(self):
        net = bp.layers.ConvTranspose2d(2, 3, kernel_size=3, padding=[(1, 1)], mode=bm.nonbatching_mode)
        self.assertEqual(net.padding, ((1, 1), (1, 1)))

    def test_padding_full_sequence(self):
        net = bp.layers.ConvTranspose2d(2, 3, kernel_size=3, padding=[(1, 1), (2, 2)],
                                        mode=bm.nonbatching_mode)
        self.assertEqual(net.padding, ((1, 1), (2, 2)))

    def test_padding_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            bp.layers.ConvTranspose2d(2, 3, kernel_size=3, padding=[(1, 1), (2, 2), (3, 3)],
                                      mode=bm.nonbatching_mode)

    def test_padding_bad_type_raises(self):
        with self.assertRaises(ValueError):
            bp.layers.ConvTranspose2d(2, 3, kernel_size=3, padding=3.5, mode=bm.nonbatching_mode)


class TestConvTransposeForwardPaths(parameterized.TestCase):
    def test_ct1d_nonbatching_with_bias(self):
        bm.random.seed()
        net = bp.layers.ConvTranspose1d(2, 3, kernel_size=3, padding=1, mode=bm.nonbatching_mode)
        y = net(jnp.ones((5, 2)))
        self.assertEqual(y.shape, (5, 3))

    def test_ct1d_nonbatching_no_bias(self):
        bm.random.seed()
        net = bp.layers.ConvTranspose1d(2, 3, kernel_size=3, b_initializer=None,
                                        mode=bm.nonbatching_mode)
        y = net(jnp.ones((5, 2)))
        self.assertEqual(y.shape, (5, 3))

    def test_ct_with_mask(self):
        bm.random.seed()
        mask = jnp.ones((3, 2, 3))
        net = bp.layers.ConvTranspose1d(2, 3, kernel_size=3, mask=mask, mode=bm.nonbatching_mode)
        y = net(jnp.ones((5, 2)))
        self.assertEqual(y.shape, (5, 3))

    def test_ct_mask_shape_mismatch_raises(self):
        net = bp.layers.ConvTranspose1d(2, 3, kernel_size=3, mask=jnp.ones((9, 9, 9, 9)),
                                        mode=bm.nonbatching_mode)
        with self.assertRaises(ValueError):
            net(jnp.ones((5, 2)))

    def test_base_transpose_repr(self):
        net = bp.layers.ConvTranspose1d(2, 3, kernel_size=3, mode=bm.nonbatching_mode)
        self.assertIn('in_channels=2', repr(net))

    def test_base_transpose_check_input_dim_not_implemented(self):
        # _GeneralConvTranspose._check_input_dim raises NotImplementedError (line 557).
        net = _GeneralConvTranspose(num_spatial_dims=1, in_channels=2, out_channels=3,
                                    kernel_size=3, mode=bm.nonbatching_mode)
        with self.assertRaises(NotImplementedError):
            net(jnp.ones((5, 2)))


class TestConvTransposeCheckInputDim(parameterized.TestCase):
    @parameterized.named_parameters(
        dict(testcase_name='CT1d', klass='ConvTranspose1d', bad=(2, 3, 4, 5), chans=(5, 9)),
        dict(testcase_name='CT2d', klass='ConvTranspose2d', bad=(2, 3), chans=(5, 5, 9)),
        dict(testcase_name='CT3d', klass='ConvTranspose3d', bad=(2, 3), chans=(5, 5, 5, 9)),
    )
    def test_bad_ndim_and_channels(self, klass, bad, chans):
        layer = getattr(bp.layers, klass)
        net = layer(2, 3, kernel_size=3, mode=bm.nonbatching_mode)
        with self.assertRaises(ValueError):
            net(jnp.ones(bad))
        with self.assertRaises(ValueError):
            net(jnp.ones(chans))


if __name__ == '__main__':
    absltest.main()
