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
from absl.testing import parameterized

import brainpy.math as bm
from brainpy.math.surrogate import _two_inputs as two_inputs


class TestTwoInputsGrad(parameterized.TestCase):
    def __init__(self, *args, platform='cpu', **kwargs):
        super(TestTwoInputsGrad, self).__init__(*args, **kwargs)
        bm.set_platform(platform)
        print()

    @parameterized.named_parameters(
        dict(testcase_name=f'{name}_x64={x64}',
             func=getattr(two_inputs, name),
             x64=x64)
        for name in two_inputs.__all__
        for x64 in [True, False]
    )
    def test_bm_grad(self, func, x64):
        if x64:
            bm.enable_x64()

        xs = bm.arange(-3, 3, 0.005)
        grads = bm.vector_grad(func)(xs[:-1], xs[1:])
        self.assertTrue(grads.size == xs.size - 1)

        if x64:
            bm.disable_x64()

    @parameterized.named_parameters(
        dict(testcase_name=f'{name}_x64={x64}',
             func=getattr(two_inputs, name),
             x64=x64, )
        for name in two_inputs.__all__
        for x64 in [True, False]
    )
    def test_jax_vjp(self, func, x64):
        if x64:
            bm.enable_x64()

        xs = bm.arange(-3, 3, 0.005)
        primals, f_vjp = jax.vjp(func, xs[:-1], xs[1:])
        grad2 = f_vjp(jax.numpy.ones(xs.size - 1))
        self.assertTrue(grad2[0].size == xs.size - 1)

        if x64:
            bm.disable_x64()
