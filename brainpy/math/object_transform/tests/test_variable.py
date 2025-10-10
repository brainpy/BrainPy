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

import pytest

import brainpy.math as bm


class TestVar(unittest.TestCase):
    def test1(self):
        class A(bm.BrainPyObject):
            def __init__(self):
                super().__init__()
                self.a = bm.Variable(1)
                self.f1 = bm.jit(self.f)
                self.f2 = bm.jit(self.ff)
                self.f3 = bm.jit(self.fff)

            def f(self):
                b = self.tracing_variable('b', bm.ones, (1,))
                self.a += (b * 2)
                return self.a.value

            def ff(self):
                self.b += 1.

            def fff(self):
                self.f()
                self.ff()
                self.b *= self.a
                return self.b.value

        with pytest.raises(NotImplementedError):
            print()
            f_jit = bm.jit(A().f)
            f_jit()

            print()
            a = A()
