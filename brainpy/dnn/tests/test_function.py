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
from absl.testing import absltest
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm


class TestFunction(parameterized.TestCase):

    def test_flatten_batching_mode(self):
        bm.random.seed()
        layer = bp.dnn.Flatten(mode=bm.BatchingMode())
        input = bm.random.randn(20, 10, 10, 6)

        output = layer.update(input)

        expected_shape = (20, 600)
        self.assertEqual(output.shape, expected_shape)

    def test_flatten_non_batching_mode(self):
        bm.random.seed()
        layer = bp.dnn.Flatten(mode=bm.NonBatchingMode())
        input = bm.random.randn(10, 10, 6)

        output = layer.update(input)

        expected_shape = (600,)
        self.assertEqual(output.shape, expected_shape)

    def test_unflatten(self):
        bm.random.seed()
        layer = bp.dnn.Unflatten(1, (10, 6), mode=bm.NonBatchingMode())
        input = bm.random.randn(5, 60)
        output = layer.update(input)
        expected_shape = (5, 10, 6)
        self.assertEqual(output.shape, expected_shape)


if __name__ == '__main__':
    absltest.main()
