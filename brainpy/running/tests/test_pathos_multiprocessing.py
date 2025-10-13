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
import sys

import jax
import pytest
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm

if sys.platform == 'win32' and sys.version_info.minor >= 11:
    pytest.skip('python 3.11 does not support.', allow_module_level=True)
else:
    pytest.skip('Cannot pass tests.', allow_module_level=True)


class TestParallel(parameterized.TestCase):
    @parameterized.product(
        duration=[1e2, 1e3, 1e4, 1e5]
    )
    def test_cpu_unordered_parallel_v1(self, duration):
        @jax.jit
        def body(inp):
            return bm.for_loop(lambda x: x + 1e-9, inp)

        input_long = bm.random.randn(1, int(duration / bm.dt), 3) / 100

        r = bp.running.cpu_ordered_parallel(body, {'inp': [input_long, input_long]}, num_process=2)
        assert bm.allclose(r[0], r[1])

    @parameterized.product(
        duration=[1e2, 1e3, 1e4, 1e5]
    )
    def test_cpu_unordered_parallel_v2(self, duration):
        @jax.jit
        def body(inp):
            return bm.for_loop(lambda x: x + 1e-9, inp)

        input_long = bm.random.randn(1, int(duration / bm.dt), 3) / 100

        r = bp.running.cpu_unordered_parallel(body, {'inp': [input_long, input_long]}, num_process=2)
        assert bm.allclose(r[0], r[1])
