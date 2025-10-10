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

import brainpy as bp
import brainpy.math as bm


class TestDiffEncoder(unittest.TestCase):
    def test_delta(self):
        a = bm.array([1, 2, 2.9, 3, 3.9])
        encoder = bp.encoding.DiffEncoder(threshold=1)
        r = encoder.multi_steps(a)
        excepted = bm.asarray([1., 1., 0., 0., 0.])
        self.assertTrue(bm.allclose(r, excepted))

        encoder = bp.encoding.DiffEncoder(threshold=1, padding=True)
        r = encoder.multi_steps(a)
        excepted = bm.asarray([0., 1., 0., 0., 0.])
        self.assertTrue(bm.allclose(r, excepted))

    def test_delta_off_spike(self):
        b = bm.array([1, 2, 0, 2, 2.9])
        encoder = bp.encoding.DiffEncoder(threshold=1, off_spike=True)
        r = encoder.multi_steps(b)
        excepted = bm.asarray([1., 1., -1., 1., 0.])
        self.assertTrue(bm.allclose(r, excepted))

        encoder = bp.encoding.DiffEncoder(threshold=1, padding=True, off_spike=True)
        r = encoder.multi_steps(b)
        excepted = bm.asarray([0., 1., -1., 1., 0.])
        self.assertTrue(bm.allclose(r, excepted))


class TestLatencyEncoder(unittest.TestCase):
    def test_latency(self):
        a = bm.array([0.02, 0.5, 1])
        encoder = bp.encoding.LatencyEncoder(method='linear')

        r = encoder.multi_steps(a, n_time=0.5)
        excepted = bm.asarray(
            [[0., 0., 1.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 1., 0.],
             ]
        )
        self.assertTrue(bm.allclose(r, excepted))

        r = encoder.multi_steps(a, n_time=1.0)
        excepted = bm.asarray(
            [[0., 0., 1.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 1., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.],
             [1., 0., 0.],
             ]
        )
        self.assertTrue(bm.allclose(r, excepted))

        encoder = bp.encoding.LatencyEncoder(method='linear', normalize=True)
        r = encoder.multi_steps(a, n_time=0.5)
        excepted = bm.asarray(
            [[0., 0., 1.],
             [0., 0., 0.],
             [0., 1., 0.],
             [0., 0., 0.],
             [1., 0., 0.],
             ]
        )
        self.assertTrue(bm.allclose(r, excepted))
