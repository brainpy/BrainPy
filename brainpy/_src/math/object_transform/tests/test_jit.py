# -*- coding: utf-8 -*-


import brainpy as bp
import brainpy.math as bm


class TestJIT(bp.testing.UnitTestCase):
  def test_jaxarray_inside_jit1(self):
    class SomeProgram(bp.BrainPyObject):
      def __init__(self):
        super(SomeProgram, self).__init__()
        self.a = bm.zeros(2)
        self.b = bm.Variable(bm.ones(2))

      def __call__(self, *args, **kwargs):
        a = bm.random.uniform(size=2)
        a = a.at[0].set(1.)
        self.b += a
        return self.b

    program = SomeProgram()
    b_out = bm.jit(program)()
    self.assertTrue(bm.array_equal(b_out, program.b))

  def test_class_jit1(self):
    class SomeProgram(bp.BrainPyObject):
      def __init__(self):
        super(SomeProgram, self).__init__()
        self.a = bm.zeros(2)
        self.b = bm.Variable(bm.ones(2))

      @bm.cls_jit
      def __call__(self):
        a = bm.random.uniform(size=2)
        a = a.at[0].set(1.)
        self.b += a
        return self.b

      @bm.cls_jit_inline
      def update(self, x):
        self.b += x

    program = SomeProgram()
    new_b = program()
    self.assertTrue(bm.allclose(new_b, program.b))
    program.update(1.)
    self.assertTrue(bm.allclose(new_b + 1., program.b))



