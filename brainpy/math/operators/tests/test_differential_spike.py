# -*- coding: utf-8 -*-


import unittest
from functools import partial

import brainpy.math as bm


def test_sp_sigmoid_grad():
  f_grad = bm.vector_grad(lambda a: bm.spike_with_sigmoid_grad(a, 1.))
  rng = bm.random.RandomState()
  x = rng.random(10) - 0.5
  print(f_grad(x))


class TestSpike2SigmoidGrad(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(TestSpike2SigmoidGrad, self).__init__(*args, **kwargs)

    @partial(bm.vector_grad, return_value=True)
    def f4(a, b):
      return b + bm.spike_with_sigmoid_grad(a + 0.1, 100.) * bm.spike_with_sigmoid_grad(-a, 100.)

    @partial(bm.vector_grad, return_value=True)
    def f5(a, b):
      return b + bm.spike2_with_sigmoid_grad(a + 0.1, a, 100.)

    self.f4 = f4
    self.f5 = f5

  def test_sp_sigmoid_grad2(self):
    a = bm.ones(10) * 2
    b = bm.ones(10)
    grad1, val1 = self.f4(a, b)
    grad2, val2 = self.f5(a, b)
    self.assertTrue(bm.array_equal(grad1, grad2))
    self.assertTrue(bm.array_equal(val1, val2))

  def test_sp_sigmoid_grad1(self):
    a = bm.zeros(10)
    b = bm.ones(10)
    grad1, val1 = self.f4(a, b)
    grad2, val2 = self.f5(a, b)
    print(grad2)
    print(grad1)

    self.assertTrue(~bm.array_equal(grad1, grad2))
    self.assertTrue(~bm.array_equal(val1, val2))

  def test_sp_sigmoid_grad3(self):
    a = bm.ones(10) * -2
    b = bm.ones(10)
    grad1, val1 = self.f4(a, b)
    grad2, val2 = self.f5(a, b)
    self.assertTrue(bm.array_equal(grad1, grad2))
    self.assertTrue(bm.array_equal(val1, val2))





