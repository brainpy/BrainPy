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

import brainpy as bp
import brainpy.math as bm


class TestSliceView(unittest.TestCase):
    def test_lif(self):
        lif = bp.neurons.LIF(10)
        lif_tile = lif[5:]
        print(lif_tile.V.shape)
        print(lif_tile.varshape)

        print('Before modification: ')
        print(lif.V)
        lif_tile.V += 10.

        self.assertTrue(bm.allclose(lif.V, bm.concatenate([bm.zeros(5), bm.ones(5) * 10.])))
        print('After modification 1: ')
        print(lif.V)

        lif_tile.V.value = bm.ones(5) * 40.
        self.assertTrue(bm.allclose(lif.V, bm.concatenate([bm.zeros(5), bm.ones(5) * 40.])))
        print('After modification 2: ')
        print(lif.V)

    def test_lif_train_mode(self):
        lif = bp.neurons.LIF(10, mode=bm.training_mode)
        lif_tile = lif[5:]
        print(lif_tile.V.shape)
        print(lif_tile.varshape)

        print('Before modification: ')
        print(lif.V)
        lif_tile.V += 10.

        self.assertTrue(bm.allclose(lif.V, bm.hstack([bm.zeros((1, 5)), bm.ones((1, 5)) * 10.])))
        print('After modification 1: ')
        print(lif.V)

        lif_tile.V.value = bm.ones((1, 5)) * 40.
        self.assertTrue(bm.allclose(lif.V, bm.hstack([bm.zeros((1, 5)), bm.ones((1, 5)) * 40.])))
        print('After modification 2: ')
        print(lif.V)
