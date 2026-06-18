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
import unittest

import jax
import numpy as np

import braintools.init as _bt_init
import brainpy as bp
import brainpy.math as bm


class TestZeroInit(unittest.TestCase):
    def test_zero_init(self):
        init = bp.init.ZeroInit()
        for size in [(100,), (10, 20), (10, 20, 30)]:
            weights = init(size)
            assert weights.shape == size
            assert isinstance(weights, bp.math.ndarray)


class TestOneInit(unittest.TestCase):
    def test_one_init(self):
        for size in [(100,), (10, 20), (10, 20, 30)]:
            for value in [0., 1., -1.]:
                init = bp.init.OneInit(value=value)
                weights = init(size)
                assert weights.shape == size
                assert (weights == value).all()


class TestIdentityInit(unittest.TestCase):
    def test_identity_init(self):
        for size in [(100,), (10, 20)]:
            for value in [0., 1., -1.]:
                init = bp.init.Identity(value=value)
                weights = init(size)
                if len(size) == 1:
                    assert weights.shape == (size[0], size[0])
                else:
                    assert weights.shape == size
                assert isinstance(weights, (bp.math.ndarray, jax.Array))


def _np(x):
    return np.asarray(x)


class TestConstant(unittest.TestCase):
    def test_scalar_value(self):
        init = bp.init.Constant(value=3.0)
        w = init((2, 4))
        self.assertEqual(w.shape, (2, 4))
        self.assertTrue((_np(w) == 3.0).all())
        self.assertIsInstance(w, bp.math.ndarray)

    def test_array_value_broadcasts(self):
        # An array value must broadcast across the trailing dimension exactly
        # like ``bm.ones(shape) * value`` did before the delegation.
        init = bp.init.Constant(value=np.array([1.0, 2.0, 3.0]))
        w = init((4, 3))
        self.assertTrue(np.allclose(_np(w), np.broadcast_to([1.0, 2.0, 3.0], (4, 3))))

    def test_dtype_propagates(self):
        w = bp.init.Constant(value=1.0)((2, 2), dtype=jax.numpy.float16)
        self.assertEqual(_np(w).dtype, np.float16)


class TestZeroInitDetails(unittest.TestCase):
    def test_values_and_dtype(self):
        w = bp.init.ZeroInit()((3, 3), dtype=jax.numpy.float16)
        self.assertTrue((_np(w) == 0).all())
        self.assertEqual(_np(w).dtype, np.float16)
        self.assertIsInstance(w, bp.math.ndarray)


class TestOneInitEqualsOnes(unittest.TestCase):
    def test_default_is_ones(self):
        w = bp.init.OneInit()((2, 3))
        self.assertTrue((_np(w) == 1.0).all())


class TestIdentityDetails(unittest.TestCase):
    def test_values_1d_is_square_scaled(self):
        w = bp.init.Identity(value=2.0)((3,))
        self.assertTrue(np.allclose(_np(w), np.eye(3) * 2.0))

    def test_values_2d_rectangular(self):
        w = bp.init.Identity(value=1.0)((2, 3))
        self.assertTrue(np.allclose(_np(w), np.eye(2, 3)))

    def test_more_than_2d_raises(self):
        with self.assertRaises(ValueError):
            bp.init.Identity()((2, 3, 4))

    def test_invalid_shape_type_raises(self):
        with self.assertRaises(ValueError):
            bp.init.Identity()(1.5)


class TestEquivalenceToBraintools(unittest.TestCase):
    """Deterministic inits must match their braintools.init counterparts."""

    def test_zero_init(self):
        for size in [(5,), (3, 4)]:
            self.assertTrue(np.allclose(_np(bp.init.ZeroInit()(size)),
                                        _np(_bt_init.ZeroInit()(size))))

    def test_constant(self):
        for size in [(5,), (3, 4)]:
            self.assertTrue(np.allclose(_np(bp.init.Constant(2.5)(size)),
                                        _np(_bt_init.Constant(2.5)(size))))

    def test_identity(self):
        # 2D shapes delegate 1:1 to braintools.init.Identity.
        for size in [(2, 3), (3, 3)]:
            self.assertTrue(np.allclose(_np(bp.init.Identity(1.5)(size)),
                                        _np(_bt_init.Identity(scale=1.5)(size))))
        # brainpy treats a 1D shape (n,) as a request for an (n, n) identity
        # matrix, so it expands to the square shape before delegating (whereas
        # braintools.init.Identity((n,)) would return a 1D vector).
        self.assertTrue(np.allclose(_np(bp.init.Identity(1.5)((4,))),
                                    _np(_bt_init.Identity(scale=1.5)((4, 4)))))
