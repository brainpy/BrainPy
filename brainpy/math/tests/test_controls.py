# -*- coding: utf-8 -*-

import unittest
import brainpy as bp
import brainpy.math as bm


class TestScan(unittest.TestCase):
  def test_easy_scan1(self):
    def make_node(v1, v2):
      def update(x):
        v1.value = v1 * x
        return (v1 + v2) * x

      return update

    bp.math.random.seed()
    _v1 = bm.Variable(bm.random.normal(size=10))
    _v2 = bm.Variable(bm.random.random(size=10))
    _xs = bm.random.uniform(size=(4, 10))

    scan_f = bm.make_loop(make_node(_v1, _v2),
                          dyn_vars=(_v1, _v2),
                          out_vars=(_v1,),
                          has_return=True)
    outs, returns = scan_f(_xs)
    for out in outs:
      print(out.shape)
    print(outs)
    print(returns.shape)
    print(returns)

    print('-' * 20)
    scan_f = bm.make_loop(make_node(_v1, _v2),
                          dyn_vars=(_v1, _v2),
                          out_vars=_v1,
                          has_return=True)
    outs, returns = scan_f(_xs)
    print(outs.shape)
    print(outs)
    print(returns.shape)
    print(returns)

    print('-' * 20)
    scan_f = bm.make_loop(make_node(_v1, _v2),
                          dyn_vars=(_v1, _v2),
                          has_return=True)
    outs, returns = scan_f(_xs)
    print(outs)
    print(returns.shape)
    print(returns)

  def test_jaxarray(self):
    def make_node(v1, v2):
      def update(x):
        v1.value = v1 * x
        return (v1 + v2) * x

      return update

    bp.math.random.seed()
    _v1 = bm.random.normal(size=10)
    _v2 = bm.random.random(size=10)
    _xs = bm.random.uniform(size=(4, 10))

    scan_f = bm.make_loop(make_node(_v1, _v2),
                          dyn_vars=(_v1, _v2),
                          out_vars=(_v1,),
                          has_return=True)
    with self.assertRaises(bp.errors.MathError):
      outs, returns = scan_f(_xs)
