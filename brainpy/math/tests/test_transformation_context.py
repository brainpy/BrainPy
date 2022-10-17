# -*- coding: utf-8 -*-


import unittest

import brainpy as bp
import brainpy.math as bm


class TestJIT(unittest.TestCase):
  def test1(self):
    @bm.jit
    def f1(a):
      a[:] = 1.
      return a

    a = bm.zeros(10)
    with self.assertRaises(bp.errors.MathError):
      print(f1(a))

  def test2(self):
    @bm.jit
    def f1(a):
      b = a + 1

      @bm.jit
      def f2(x):
        x.value = 1.
        return x

      return f2(b)

    with self.assertRaises(bp.errors.MathError):
      print(f1(bm.ones(2)))

  def test3(self):
    @bm.jit
    def f1(a):
      return a + 1

    @bm.jit
    def f2(b):
      b[:] = 1.
      return b

    with self.assertRaises(bp.errors.MathError):
      print(f2(f1(bm.ones(2))))

  def test4(self):
    @bm.jit
    def f2(a):
      b = bm.ones(1)
      b += 10
      return a + b

    print(f2(bm.ones(1)))
