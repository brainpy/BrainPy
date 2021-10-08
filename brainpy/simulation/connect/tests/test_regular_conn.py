# -*- coding: utf-8 -*-

import brainpy as bp
from brainpy import connect


def test_one2one():
  for size in [100, (3, 4), (4, 5, 6)]:
    conn = connect.One2One()(pre_size=size, post_size=size)
    mat = conn.require(connect.CONN_MAT)
    res = conn.require('pre_ids', 'post_ids', 'pre2post', 'pre_slice')
    num = bp.simulation.size2len(size)

    actual_mat = bp.math.zeros((num, num), dtype=bp.math.bool_)
    actual_mat = bp.math.fill_diagonal(actual_mat, True)
    assert bp.math.array_equal(actual_mat, mat)
    assert bp.math.array_equal(res[0], bp.math.arange(num))
    assert bp.math.array_equal(res[1], bp.math.arange(num))

    print()
    print('pre_ids', res[0])
    print('post_ids', res[1])
    print('pre2post', res[2])
    print('pre_slice', res[3])


def test_all2all():
  for has_self in [True, False]:
    for size in [100, (3, 4), (4, 5, 6)]:
      conn = connect.All2All(include_self=has_self)(pre_size=size, post_size=size)
      mat = conn.require(connect.CONN_MAT)
      res = conn.require('pre_ids', 'post_ids', 'pre2post', 'pre_slice')
      num = bp.simulation.size2len(size)

      actual_mat = bp.math.ones((num, num), dtype=bp.math.bool_)
      if not has_self:
        actual_mat = bp.math.fill_diagonal(actual_mat, False)
      assert bp.math.array_equal(actual_mat, mat)

      print()
      print('pre_ids', res[0])
      print('post_ids', res[1])
      print('pre2post', res[2])
      print('pre_slice', res[3])


def test_grid_four():
  pass

