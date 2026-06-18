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
import pytest

import brainpy as bp
import brainpy.math as bm


class TestJIT(unittest.TestCase):
    def test_jaxarray_inside_jit1(self):
        # Ensure clean state before test
        bm.random.seed(123)

        class SomeProgram(bp.BrainPyObject):
            def __init__(self):
                super(SomeProgram, self).__init__()
                self.a = bm.zeros(2)
                self.b = bm.Variable(bm.ones(2))

            def __call__(self, *args, **kwargs):
                a = bm.random.uniform(size=2)
                a = a.at[0].set(1.)
                self.b += a
                return self.b.value

        program = SomeProgram()
        b_out = bm.jit(program)()
        self.assertTrue(bm.array_equal(b_out, program.b))

    def test_jaxarray_inside_jit1_disable(self):
        # Ensure clean state before test
        bm.random.seed(123)

        class SomeProgram(bp.BrainPyObject):
            def __init__(self):
                super(SomeProgram, self).__init__()
                self.a = bm.zeros(2)
                self.b = bm.Variable(bm.ones(2))

            def __call__(self, *args, **kwargs):
                a = bm.random.uniform(size=2)
                a = a.at[0].set(1.)
                self.b += a
                return self.b.value

        program = SomeProgram()
        with jax.disable_jit():
            b_out = bm.jit(program)()
            self.assertTrue(bm.array_equal(b_out, program.b))
            print(b_out)

    def test_jit_with_static(self):
        a = bm.Variable(bm.ones(2))

        @bm.jit(static_argnums=1)
        def f(b, c):
            a.value *= b
            a.value /= c

        f(1., 2.)
        self.assertTrue(bm.allclose(a.value, 0.5))

        @bm.jit(static_argnames=['c'])
        def f2(b, c):
            a.value *= b
            a.value /= c

        f2(2., c=1.)
        self.assertTrue(bm.allclose(a.value, 1.))


class TestClsJIT(unittest.TestCase):

    @pytest.mark.skip(reason="not implemented")
    def test_class_jit1(self):
        # Ensure clean state before test
        import jax
        import gc

        # Clear all caches and state
        jax.clear_caches()
        gc.collect()

        # Reset random state
        bm.random.seed(123)

        class SomeProgram(bp.BrainPyObject):
            def __init__(self):
                super(SomeProgram, self).__init__()
                self.a = bm.zeros(2)
                self.b = bm.Variable(bm.ones(2))

            @bm.cls_jit
            def __call__(self):
                a = bm.random.uniform(size=2)
                a = a.at[0].set(1.)
                self.b.value += a
                return self.b.value

            @bm.cls_jit(inline=True)
            def update(self, x):
                self.b.value += x

        program = SomeProgram()
        new_b = program()
        self.assertTrue(bm.allclose(new_b, program.b))
        program.update(1.)
        self.assertTrue(bm.allclose(new_b + 1., program.b))

    def test_class_jit2(self):
        # Ensure clean state before test
        bm.random.seed(123)

        class SomeProgram(bp.BrainPyObject):
            def __init__(self):
                super(SomeProgram, self).__init__()
                self.a = bm.zeros(2)
                self.b = bm.Variable(bm.ones(2))

                self.call1 = bm.jit(self.call, static_argnums=0)
                self.call2 = bm.jit(self.call, static_argnames=['fit'])

            def call(self, fit=True):
                a = bm.random.uniform(size=2)
                if fit:
                    a = a.at[0].set(1.)
                self.b.value += a
                return self.b.value

        program = SomeProgram()
        new_b1 = program.call1(True)
        new_b2 = program.call2(fit=False)
        print()
        print(new_b1, )
        print(new_b2, )
        with self.assertRaises(jax.errors.TracerBoolConversionError):
            new_b3 = program.call2(False)

    @pytest.mark.skip(reason="not implemented")
    def test_class_jit1_with_disable(self):
        # Ensure clean state before test
        bm.random.seed(123)

        class SomeProgram(bp.BrainPyObject):
            def __init__(self):
                super(SomeProgram, self).__init__()
                self.a = bm.zeros(2)
                self.b = bm.Variable(bm.ones(2))

            @bm.cls_jit
            def __call__(self):
                a = bm.random.uniform(size=2)
                a = a.at[0].set(1.)
                self.b.value += a
                return self.b.value

            @bm.cls_jit(inline=True)
            def update(self, x):
                self.b.value += x

        program = SomeProgram()
        with jax.disable_jit():
            new_b = program()
            self.assertTrue(bm.allclose(new_b, program.b))
        with jax.disable_jit():
            program.update(1.)
            self.assertTrue(bm.allclose(new_b + 1., program.b))

    def test_cls_jit_with_static(self):
        class MyObj:
            def __init__(self):
                self.a = bm.Variable(bm.ones(2))

            @bm.cls_jit(static_argnums=0)
            def f(self, b, c):
                self.a.value *= b
                self.a.value /= c

        obj = MyObj()
        obj.f(1., 2.)
        self.assertTrue(bm.allclose(obj.a.value, 0.5))

        class MyObj2:
            def __init__(self):
                self.a = bm.Variable(bm.ones(2))

            @bm.cls_jit(static_argnames=['c'])
            def f(self, b, c):
                self.a.value *= b
                self.a.value /= c

        obj = MyObj2()
        obj.f(1., c=2.)
        self.assertTrue(bm.allclose(obj.a.value, 0.5))

# class TestDebug(unittest.TestCase):
#   def test_debug1(self):
#     a = bm.random.RandomState()
#
#     @bm.jit
#     def f(b):
#       print(a.value)
#       return a + b + a.random()
#
#     f(1.)
#
#     with jax.disable_jit():
#       f(1.)
#
#   def test_print_info1(self):
#     file = tempfile.TemporaryFile(mode='w+')
#
#     @bm.jit
#     def f2(a, b):
#       print('compiling f2 ...', file=file)
#       return a + b
#
#     @bm.jit
#     def f1(a):
#       print('compiling f1 ...', file=file)
#       return f2(a, 1.)
#
#     expect_res = '''
# compiling f1 ...
# compiling f2 ...
# compiling f1 ...
# compiling f2 ...
#     '''
#     self.assertTrue(f1(1.) == 2.)
#     file.seek(0)
#     self.assertTrue(file.read().strip() == expect_res.strip())
#
#     file = tempfile.TemporaryFile(mode='w+')
#     with jax.disable_jit():
#       expect_res = '''
# compiling f1 ...
# compiling f2 ...
#       '''
#       self.assertTrue(f1(1.) == 2.)
#       file.seek(0)
#       self.assertTrue(file.read().strip() == expect_res.strip())
#
#   def test_print_info2(self):
#     file = tempfile.TemporaryFile(mode='w+')
#
#     @bm.jit
#     def f1(a):
#       @bm.jit
#       def f2(a, b):
#         print('compiling f2 ...', file=file)
#         return a + b
#
#       print('compiling f1 ...', file=file)
#       return f2(a, 1.)
#
#     expect_res = '''
# compiling f1 ...
# compiling f2 ...
# compiling f1 ...
# compiling f2 ...
# compiling f2 ...
#     '''
#     self.assertTrue(f1(1.) == 2.)
#     file.seek(0)
#     self.assertTrue(file.read().strip() == expect_res.strip())
#
#     file = tempfile.TemporaryFile(mode='w+')
#     with jax.disable_jit():
#       expect_res = '''
# compiling f1 ...
# compiling f2 ...
#       '''
#       self.assertTrue(f1(1.) == 2.)
#       file.seek(0)
#       # print(file.read().strip())
#       self.assertTrue(file.read().strip() == expect_res.strip())
