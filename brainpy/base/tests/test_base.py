# -*- coding: utf-8 -*-

import unittest

import brainpy as bp
import brainpy.math as bm


class TestCollectionFunction(unittest.TestCase):
  def test_f_nodes(self):
    class C(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(C, self).__init__()

    class B(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(B, self).__init__()

        self.child1 = C()
        self.child2 = C()

    class A(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(A, self).__init__()

        self.child1 = B()
        self.child2 = B()

    net = bp.dyn.Network(a1=A(), a2=A())
    print(net.nodes(level=2))
    self.assertTrue(len(net.nodes(level=0)) == 0)
    self.assertTrue(len(net.nodes(level=0, include_self=False)) == 0)
    self.assertTrue(len(net.nodes(level=1)) == (1 + 2))
    self.assertTrue(len(net.nodes(level=1, include_self=False)) == 2)
    self.assertTrue(len(net.nodes(level=2)) == (1 + 2 + 4))
    self.assertTrue(len(net.nodes(level=2, include_self=False)) == (2 + 4))
    self.assertTrue(len(net.nodes(level=3)) == (1 + 2 + 4 + 8))
    self.assertTrue(len(net.nodes(level=3, include_self=False)) == (2 + 4 + 8))

  def test_f_vars(self):
    class C(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(C, self).__init__()

        self.var1 = bm.Variable(bm.zeros(1))
        self.var2 = bm.Variable(bm.zeros(1))

    class B(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(B, self).__init__()

        self.child1 = C()
        self.child2 = C()

        self.var1 = bm.Variable(bm.zeros(1))
        self.var2 = bm.Variable(bm.zeros(1))

    class A(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(A, self).__init__()

        self.child1 = B()
        self.child2 = B()

        self.var1 = bm.Variable(bm.zeros(1))
        self.var2 = bm.Variable(bm.zeros(1))

    net = bp.dyn.Network(a1=A(), a2=A())
    print(net.vars(level=2))
    self.assertTrue(len(net.vars(level=0)) == 0)
    self.assertTrue(len(net.vars(level=0, include_self=False)) == 0)
    self.assertTrue(len(net.vars(level=1)) == 2*2)
    self.assertTrue(len(net.vars(level=1, include_self=False)) == 2*2)
    self.assertTrue(len(net.vars(level=2)) == (2 + 4) * 2)
    self.assertTrue(len(net.vars(level=2, include_self=False)) == (2 + 4) * 2)
    self.assertTrue(len(net.vars(level=3)) == (2 + 4 + 8) * 2)
    self.assertTrue(len(net.vars(level=3, include_self=False)) == (2 + 4 + 8) * 2)





