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
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten

import brainpy.math as bm
from brainpy.math import Variable


class TestJaxArray(unittest.TestCase):
    def test_tree(self):
        structured = {'a': Variable(jnp.zeros(1)),
                      'b': (Variable(jnp.ones(2)),
                            Variable(jnp.ones(2) * 2))}
        flat, tree = tree_flatten(structured)
        unflattened = tree_unflatten(tree, flat)
        print("\nstructured={}\n\n  flat={}\n\n  tree={}\n\n  unflattened={}".format(
            structured, flat, tree, unflattened))

    def test_none(self):
        # https://github.com/PKU-NIP-Lab/BrainPy/issues/144
        a = None
        b = bm.zeros(10)
        with self.assertRaises(TypeError):
            bb = a + b

        c = bm.Variable(bm.zeros(10))
        with self.assertRaises(TypeError):
            cc = a + c

        d = bm.Parameter(bm.zeros(10))
        with self.assertRaises(TypeError):
            dd = a + d

        e = bm.TrainVar(bm.zeros(10))
        with self.assertRaises(TypeError):
            ee = a + e

    def test_operation_with_numpy_array(self):
        rng = bm.random.RandomState(123)
        add = lambda: bm.asarray(rng.rand(10)) + np.zeros(1)
        # self.assertTrue(isinstance(add(), bm.Array))
        # self.assertTrue(isinstance(bm.jit(add)(), bm.Array))


class TestTracerError(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.a = bm.zeros((10, 2))
        self.f = jax.jit(self._f)

    def _f(self, b):
        self.a[:] = bm.zeros_like(self.a)
        return b + 1.

    def test_tracing(self):
        print(self.f(1.))
        with self.assertRaises(jax.errors.UnexpectedTracerError):
            print(self.f(bm.ones(10)))


class TestVariable(unittest.TestCase):
    def test_variable_init(self):
        self.assertTrue(
            bm.array_equal(bm.Variable(bm.zeros(10)),
                           bm.Variable(10))
        )
        bm.random.seed(123)
        self.assertTrue(
            not bm.array_equal(bm.Variable(bm.random.rand(10)),
                               bm.Variable(10))
        )


class TestVariableView(unittest.TestCase):
    def test_update(self):
        bm.random.seed()

        origin = bm.Variable(bm.zeros(10))
        view = bm.VariableView(origin, slice(0, 5, None))

        view.update(bm.ones(5))
        self.assertTrue(
            bm.array_equal(origin, bm.concatenate([bm.ones(5), bm.zeros(5)]))
        )

        view.value = bm.arange(5.)
        self.assertTrue(
            bm.array_equal(origin, bm.concatenate([bm.arange(5), bm.zeros(5)]))
        )

        view += 10
        self.assertTrue(
            bm.array_equal(origin, bm.concatenate([bm.arange(10, 15), bm.zeros(5)]))
        )

        bm.random.shuffle(view)
        print(view)
        print(origin)

        view.sort()
        self.assertTrue(
            bm.array_equal(origin, bm.concatenate([bm.arange(5) + 10, bm.zeros(5)]))
        )

        self.assertTrue(view.sum() == bm.sum(bm.arange(5) + 10))


class TestArrayPriority(unittest.TestCase):
    def test1(self):
        a = bm.Array(bm.zeros(10))
        assert isinstance(a + bm.ones(1).value, jax.Array)
        assert isinstance(a + np.ones(1), jax.Array)
        assert isinstance(a * np.ones(1), jax.Array)
        assert isinstance(np.ones(1) + a, jax.Array)
        assert isinstance(np.ones(1) * a, jax.Array)
        b = bm.Variable(bm.zeros(10))
        assert isinstance(b + bm.ones(1).value, jax.Array)
        assert isinstance(b + np.ones(1), jax.Array)
