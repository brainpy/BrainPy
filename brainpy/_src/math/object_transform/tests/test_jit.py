# -*- coding: utf-8 -*-
import jax

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

  def test_jaxarray_inside_jit1_disable(self):
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


class TestClsJIT(bp.testing.UnitTestCase):

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

      @bm.cls_jit(inline=True)
      def update(self, x):
        self.b += x

    program = SomeProgram()
    new_b = program()
    self.assertTrue(bm.allclose(new_b, program.b))
    program.update(1.)
    self.assertTrue(bm.allclose(new_b + 1., program.b))

  def test_class_jit1_with_disable(self):
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
        return self.b.value

      @bm.cls_jit(inline=True)
      def update(self, x):
        self.b += x

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

      @bm.cls_jit(static_argnums=1)
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

