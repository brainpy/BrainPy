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

import brainstate.environ
import jax.tree_util

import brainpy as bp
import brainpy.math as bm


class TestCollectionFunction(unittest.TestCase):
    def test_f_nodes(self):
        class C(bp.DynamicalSystem):
            def __init__(self):
                super(C, self).__init__()

        class B(bp.DynamicalSystem):
            def __init__(self):
                super(B, self).__init__()

                self.child1 = C()
                self.child2 = C()

        class A(bp.DynamicalSystem):
            def __init__(self):
                super(A, self).__init__()

                self.child1 = B()
                self.child2 = B()

        net = bp.Network(a1=A(), a2=A())
        print(net.nodes(level=2))
        self.assertTrue(len(net.nodes(level=0)) == 1)
        self.assertTrue(len(net.nodes(level=0, include_self=False)) == 0)
        self.assertTrue(len(net.nodes(level=1)) == (1 + 2))
        self.assertTrue(len(net.nodes(level=1, include_self=False)) == 2)
        self.assertTrue(len(net.nodes(level=2)) == (1 + 2 + 4))
        self.assertTrue(len(net.nodes(level=2, include_self=False)) == (2 + 4))
        self.assertTrue(len(net.nodes(level=3)) == (1 + 2 + 4 + 8))
        self.assertTrue(len(net.nodes(level=3, include_self=False)) == (2 + 4 + 8))

    def test_f_vars(self):
        class C(bp.DynamicalSystem):
            def __init__(self):
                super(C, self).__init__()

                self.var1 = bm.Variable(bm.zeros(1))
                self.var2 = bm.Variable(bm.zeros(1))

        class B(bp.DynamicalSystem):
            def __init__(self):
                super(B, self).__init__()

                self.child1 = C()
                self.child2 = C()

                self.var1 = bm.Variable(bm.zeros(1))
                self.var2 = bm.Variable(bm.zeros(1))

        class A(bp.DynamicalSystem):
            def __init__(self):
                super(A, self).__init__()

                self.child1 = B()
                self.child2 = B()

                self.var1 = bm.Variable(bm.zeros(1))
                self.var2 = bm.Variable(bm.zeros(1))

        net = bp.Network(a1=A(), a2=A())
        print(net.vars(level=2))
        self.assertTrue(len(net.vars(level=0)) == 0)
        self.assertTrue(len(net.vars(level=0, include_self=False)) == 0)
        self.assertTrue(len(net.vars(level=1)) == 2 * 2)
        self.assertTrue(len(net.vars(level=1, include_self=False)) == 2 * 2)
        self.assertTrue(len(net.vars(level=2)) == (2 + 4) * 2)
        self.assertTrue(len(net.vars(level=2, include_self=False)) == (2 + 4) * 2)
        self.assertTrue(len(net.vars(level=3)) == (2 + 4 + 8) * 2)
        self.assertTrue(len(net.vars(level=3, include_self=False)) == (2 + 4 + 8) * 2)


class TestNodeList(unittest.TestCase):
    def test_NodeList_1(self):
        bm.random.seed()

        class Object(bp.DynamicalSystem):
            def __init__(self):
                super().__init__()

                self.l1 = bp.layers.Dense(5, 10)
                self.ls = bm.NodeList([bp.layers.Dense(10, 4),
                                       bp.layers.Activation(bm.tanh),
                                       bp.layers.Dropout(0.1),
                                       bp.layers.Dense(4, 5),
                                       bp.layers.Activation(bm.relu)])

            def update(self, x):
                x = self.l1(x)
                for l in self.ls:
                    x = l(x)
                return x

        with bm.environment(mode=bm.NonBatchingMode()):
            obj = Object()
            self.assertTrue(len(obj.vars()) == 0)
            self.assertTrue(len(obj.nodes()) == 7)

            print(obj.nodes().keys())
            print("obj.nodes(method='relative'): ", obj.nodes(method='relative').keys())
            # print(jax.tree_util.tree_structure(obj))

        with bm.environment(mode=bm.TrainingMode()):
            obj = Object()
            self.assertTrue(len(obj.vars()) == 6)
            self.assertTrue(len(obj.nodes()) == 7)

            print(obj.nodes().keys())
            print("obj.nodes(method='relative'): ", obj.nodes(method='relative').keys())
            # print(jax.tree_util.tree_structure(obj))


class TestNodeDict(unittest.TestCase):
    def test_NodeDict_1(self):
        bm.random.seed()

        class Object(bp.DynamicalSystem):
            def __init__(self):
                super().__init__()

                self.l1 = bp.layers.Dense(5, 10)
                self.ls = bm.NodeDict(
                    {
                        'l1': bp.layers.Dense(10, 4),
                        'l2': bp.layers.Activation(bm.tanh),
                        'l3': bp.layers.Dropout(0.1),
                        'l4': bp.layers.Dense(4, 5),
                        'l5': bp.layers.Activation(bm.relu)
                    }
                )

            def update(self, x):
                x = self.l1(x)
                for l in self.ls:
                    x = l(x)
                return x

        with bm.environment(mode=bm.NonBatchingMode()):
            obj = Object()

            self.assertTrue(len(obj.vars()) == 0)
            self.assertTrue(len(obj.nodes()) == 7)
            self.assertTrue(len(jax.tree_util.tree_leaves(obj)) == 1)

            print(obj.nodes().keys())
            print("obj.nodes(method='relative'): ",
                  obj.nodes(method='relative').keys())
            # print(jax.tree_util.tree_structure(obj))

        with bm.environment(mode=bm.TrainingMode()):
            obj = Object()
            self.assertTrue(len(obj.vars()) == 6)
            self.assertTrue(len(obj.nodes()) == 7)

            print(obj.nodes().keys())
            print("obj.nodes(method='relative'): ",
                  obj.nodes(method='relative').keys())
            # print(jax.tree_util.tree_structure(obj))


class TestVarList(unittest.TestCase):
    def setUp(self):
        brainstate.environ.set(precision=32)

    def test_ListVar_1(self):
        bm.random.seed()

        class Object(bp.DynamicalSystem):
            def __init__(self):
                super().__init__()
                self.vs = bm.VarList([bm.Variable(1.),
                                      bm.Variable(2.),
                                      bm.Variable(bm.ones(10))])

            def update(self):
                self.vs[0] += 10.
                self.vs[1] += 10.
                self.vs[2] += 10.

        obj = Object()
        self.assertTrue(len(obj.vars()) == 3)
        self.assertTrue(len(obj.nodes()) == 1)

        @bm.jit
        def f2():
            obj()

        f2()
        print(obj.vs)
        self.assertTrue(obj.vs[0].value == 11.)
        self.assertTrue(obj.vs[1].value == 12.)
        self.assertTrue(bm.allclose(obj.vs[2].value, bm.ones(10) * 11.))


class TestVarDict(unittest.TestCase):
    def test_DictVar_1(self):
        bm.random.seed()

        class Object(bp.DynamicalSystem):
            def __init__(self):
                super().__init__()
                self.vs = bm.VarDict({'a': bm.Variable(1.),
                                      'b': bm.Variable(2.),
                                      'c': bm.Variable(bm.ones(10))})

            def update(self):
                self.vs['a'] += 10.
                self.vs['b'] += 10.
                self.vs['c'] += 10.

        obj = Object()
        print(obj.vars())
        self.assertTrue(len(obj.vars()) == 3)
        self.assertTrue(len(obj.nodes()) == 1)

        @bm.jit
        def f1():
            obj()

        f1()
        print(obj.vs)
        self.assertTrue(obj.vs['a'].value == 11.)
        self.assertTrue(obj.vs['b'].value == 12.)
        self.assertTrue(bm.allclose(obj.vs['c'].value, bm.ones(10) * 11.))


class TestRegisterBPObjectAsPyTree(unittest.TestCase):
    def test1(self):
        bm.set(bp_object_as_pytree=True)

        hh = bp.dyn.HH(1)
        hh.reset()

        tree = jax.tree.structure(hh)
        leaves = jax.tree.leaves(hh)
        # tree = jax.tree.structure(hh)
        # leaves = jax.tree.leaves(hh)

        print(tree)
        print(leaves)
        print(jax.tree.unflatten(tree, leaves))
        # print(jax.tree.unflatten(tree, leaves))
        print()


class TestStateSavingAndLoading(unittest.TestCase):
    def test_load_states(self):
        class Object(bp.DynamicalSystem):
            def __init__(self):
                super().__init__()

                self.l1 = bp.layers.Dense(5, 10)
                self.ls = bm.NodeList([bp.layers.Dense(10, 4),
                                       bp.layers.Activation(bm.tanh),
                                       bp.layers.Dropout(0.1),
                                       bp.layers.Dense(4, 5),
                                       bp.layers.Activation(bm.relu)])
                self.lif = bp.dyn.LifRef(5)

            def update(self, x):
                x = self.l1(x)
                for l in self.ls:
                    x = l(x)
                return x

        with bm.training_environment():
            obj = Object()
            variables = {k: dict(n.vars()) for k, n in obj.nodes(include_self=False).items()}
            variables = {k: v for k, v in variables.items() if len(v) > 0}

            all_states = obj.state_dict()
            all_states = {k: v for k, v in all_states.items() if len(v) > 0}
            print(set(all_states.keys()))
            print(set(variables.keys()))

            def not_close(x, y):
                assert not bm.allclose(x, y)

            def all_close(x, y):
                assert bm.allclose(x, y)

            jax.tree.map(all_close, all_states, variables, is_leaf=bm.is_bp_array)
            # jax.tree.map(all_close, all_states, variables, is_leaf=bm.is_bp_array)

            random_state = jax.tree.map(bm.random.rand_like, all_states, is_leaf=bm.is_bp_array)
            jax.tree.map(not_close, random_state, variables, is_leaf=bm.is_bp_array)
            # random_state = jax.tree.map(bm.random.rand_like, all_states, is_leaf=bm.is_bp_array)
            # jax.tree.map(not_close, random_state, variables, is_leaf=bm.is_bp_array)

            obj.load_state_dict(random_state)
            jax.tree.map(all_close, random_state, variables, is_leaf=bm.is_bp_array)
            # jax.tree.map(all_close, random_state, variables, is_leaf=bm.is_bp_array)
