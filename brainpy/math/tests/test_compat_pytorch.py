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

import brainpy.math as bm
import brainpy.math.compat_pytorch as torch
from brainpy.math import compat_pytorch


class TestFlatten(unittest.TestCase):
    def test1(self):
        rng = bm.random.default_rng(113)
        arr = rng.rand(3, 4, 5)
        a2 = compat_pytorch.flatten(arr, 1, 2)
        self.assertTrue(a2.shape == (3, 20))
        a2 = compat_pytorch.flatten(arr, 0, 1)
        self.assertTrue(a2.shape == (12, 5))

    def test2(self):
        rng = bm.random.default_rng(234)
        arr = rng.rand()
        self.assertTrue(arr.ndim == 0)
        arr = compat_pytorch.flatten(arr)
        self.assertTrue(arr.ndim == 1)


class TestUnsqueeze(unittest.TestCase):
    def test1(self):
        rng = bm.random.default_rng(999)
        arr = rng.rand(3, 4, 5)
        a = compat_pytorch.unsqueeze(arr, 0)
        self.assertTrue(a.shape == (1, 3, 4, 5))
        a = compat_pytorch.unsqueeze(arr, -3)
        self.assertTrue(a.shape == (3, 1, 4, 5))


class TestExpand(unittest.TestCase):
    def test1(self):
        rng = bm.random.default_rng(121)
        arr = rng.rand(1, 4, 5)
        a = compat_pytorch.Tensor(arr)
        a = a.expand(1, 6, 4, -1)
        self.assertTrue(a.shape == (1, 6, 4, 5))


class TestMathOperators(unittest.TestCase):
    def test_abs(self):
        arr = compat_pytorch.Tensor([-1, -2, 3])
        a = compat_pytorch.abs(arr)
        res = compat_pytorch.Tensor([1, 2, 3])
        b = compat_pytorch.absolute(arr)
        self.assertTrue(bm.array_equal(a, res))
        self.assertTrue(bm.array_equal(b, res))

    def test_add(self):
        a = compat_pytorch.Tensor([0.0202, 1.0985, 1.3506, -0.6056])
        a = compat_pytorch.add(a, 20)
        res = compat_pytorch.Tensor([20.0202, 21.0985, 21.3506, 19.3944])
        self.assertTrue(bm.array_equal(a, res))
        b = compat_pytorch.Tensor([-0.9732, -0.3497, 0.6245, 0.4022])
        c = compat_pytorch.Tensor([[0.3743], [-1.7724], [-0.5811], [-0.8017]])
        b = compat_pytorch.add(b, c, alpha=10)
        self.assertTrue(b.shape == (4, 4))
        print("b:", b)

    def test_addcdiv(self):
        rng = bm.random.default_rng(999)
        t = rng.rand(1, 3)
        t1 = rng.randn(3, 1)
        rng = bm.random.default_rng(199)
        t2 = rng.randn(1, 3)
        res = torch.addcdiv(t, t1, t2, value=0.1)
        print("t + t1/t2 * value:", res)
        res = torch.addcmul(t, t1, t2, value=0.1)
        print("t + t1*t2 * value:", res)
