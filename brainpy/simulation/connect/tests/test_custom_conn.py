# -*- coding: utf-8 -*-

import pytest

import brainpy as bp


def test_IJConn():
  conn = bp.connect.IJConn(i=bp.math.array([0, 1, 2]),
                           j=bp.math.array([0, 0, 0]))
  conn = conn(pre_size=5, post_size=3)

  print(conn.requires('pre2post'))
  print(conn.requires(bp.connect.CONN_MAT))


def test_MatConn1():
  conn = bp.connect.MatConn(conn_mat=bp.math.random.randint(2, size=(5, 3), dtype=bp.math.bool_))
  conn = conn(pre_size=5, post_size=3)

  print(conn.requires('pre2post'))
  print(conn.requires(bp.connect.CONN_MAT))


def test_MatConn2():
  conn = bp.connect.MatConn(conn_mat=bp.math.random.randint(2, size=(5, 3), dtype=bp.math.bool_))
  with pytest.raises(AssertionError):
    conn = conn(pre_size=5, post_size=1)
