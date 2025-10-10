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

import numpy as np

import brainpy as bp


class TestCaputoL1(unittest.TestCase):
    def test1(self):
        bp.math.random.seed()
        bp.math.enable_x64()
        alpha = 0.9
        intg = bp.fde.CaputoL1Schema(lambda a, t: a,
                                     alpha=alpha,
                                     num_memory=10,
                                     inits=[1., ])
        for N in [2, 3, 4, 5, 6, 7, 8]:
            diff = np.random.rand(N - 1, 1)
            memory_trace = 0
            for i in range(N - 1):
                c = (N - i) ** (1 - alpha) - (N - i - 1) ** (1 - alpha)
                memory_trace += c * diff[i]

            intg.idx[0] = N - 1
            intg.diff_states['a_diff'][:N - 1] = bp.math.asarray(diff)
            idx = ((intg.num_memory - intg.idx) + np.arange(intg.num_memory)) % intg.num_memory
            memory_trace2 = intg.coef[idx, 0] @ intg.diff_states['a_diff']

            print()
            print(memory_trace[0], )
            print(memory_trace2[0], bp.math.array_equal(memory_trace[0], memory_trace2[0]))

        bp.math.disable_x64()
