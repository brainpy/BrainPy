# -*- coding: utf-8 -*-

import brainpy as bp
from brainpy import connect


def test_one2one():
  for size in [100, (3, 4), (4, 5, 6)]:
    conn = connect.One2One()(pre_size=size, post_size=size)

    mat = conn.require(connect.CONN_MAT)
    conn.require('conn_mat', 'pre_ids', 'post_ids', 'pre2post', 'pre2syn', 'post2pre', 'post2syn')

    pre_ids = conn.pre_ids
    post_ids = conn.post_ids
    pre2post = conn.pre2post
    post2pre = conn.post2pre
    pre2syn = conn.pre2syn
    post2syn = conn.post2syn
    dense_mat = conn.dense_mat
    num = bp.tools.size2num(size)

    actual_mat = bp.math.zeros((num, num), dtype=bp.math.bool_)
    actual_mat = bp.math.fill_diagonal(actual_mat, True)
    assert bp.math.array_equal(actual_mat, dense_mat)
    assert bp.math.array_equal(pre_ids, bp.math.arange(num))
    assert bp.math.array_equal(post_ids, bp.math.arange(num))

    print()
    print('mat', dense_mat)
    print('pre_ids', pre_ids)
    print('post_ids', post_ids)
    print('pre2post', pre2post)
    print('post2pre', post2pre)
    print('pre2syn', pre2syn)
    print('post2syn', post2syn)


def test_all2all():
  for has_self in [True, False]:
    for size in [100, (3, 4), (4, 5, 6)]:
      conn = connect.All2All(include_self=has_self)(pre_size=size, post_size=size)
      mat = conn.require(connect.CONN_MAT)
      conn.require('conn_mat', 'pre_ids', 'post_ids', 'pre2post', 'pre2syn', 'post2pre', 'post2syn')
      num = bp.tools.size2num(size)

      print(mat)
      actual_mat = bp.math.ones((num, num), dtype=bp.math.bool_)
      if not has_self:
        actual_mat = bp.math.fill_diagonal(actual_mat, False)
      assert bp.math.array_equal(actual_mat, mat)

      print()
      print('mat', conn.dense_mat)
      print('pre_ids', conn.pre_ids)
      print('post_ids', conn.post_ids)
      print('pre2post', conn.pre2post)
      print('post2pre', conn.post2pre)
      print('pre2syn', conn.pre2syn)
      print('post2syn', conn.post2syn)


def test_grid_four():
  pass

