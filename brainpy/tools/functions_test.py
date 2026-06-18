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


class TestFunction(unittest.TestCase):
    def test_compose(self):
        f = lambda a: a + 1
        g = lambda a: a * 10
        fun1 = bp.tools.compose(f, g)
        fun2 = bp.tools.pipe(g, f)

        arr = bm.random.randn(10)
        r1 = fun1(arr)
        r2 = fun2(arr)
        groundtruth = f(g(arr))
        self.assertTrue(bm.allclose(r1, r2))
        self.assertTrue(bm.allclose(r1, groundtruth))
