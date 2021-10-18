# -*- coding: utf-8 -*-


import brainpy.math.jax as bm


def test_easy_scan1():

  def make_node(v1, v2):
    def update(x):
      v1.value = v1 * x
      return (v1 + v2) * x

    return update

  _v1 = bm.random.normal(size=10)
  _v2 = bm.random.random(size=10)
  _xs = bm.random.uniform(size=(4, 10))

  scan_f = bm.easy_scan(make_node(_v1, _v2),
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
  scan_f = bm.easy_scan(make_node(_v1, _v2),
                        dyn_vars=(_v1, _v2),
                        out_vars=_v1,
                        has_return=True)
  outs, returns = scan_f(_xs)
  print(outs.shape)
  print(outs)
  print(returns.shape)
  print(returns)

  print('-' * 20)
  scan_f = bm.easy_scan(make_node(_v1, _v2),
                        dyn_vars=(_v1, _v2),
                        has_return=True)
  outs, returns = scan_f(_xs)
  print(outs)
  print(returns.shape)
  print(returns)

