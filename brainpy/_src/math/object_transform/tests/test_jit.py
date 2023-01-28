# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm
import unittest


class TestJaxArrayJIT(unittest.TestCase):

  def test_jaxarray_inside_jit1(self):
    bp.math.random.seed()

    class SomeProgram(bp.BrainPyObject):
      def __init__(self):
        super(SomeProgram, self).__init__()
        self.a = bm.zeros(2)
        self.b = bm.Variable(bm.ones(2))

      def __call__(self, *args, **kwargs):
        a = bm.random.uniform(size=2)
        a = a.at[0].set(1.)
        self.b += a

    run = bm.jit(SomeProgram())
    run()
