# -*- coding: utf-8 -*-

import pytest

import brainpy as bp


def test_random_prob_numpy():
  for bk in ['numpy', 'jax']:
    bp.math.use_backend(bk)

    conn1 = bp.connect.FixedProb(prob=0.1, seed=123)
    conn1(pre_size=(10, 20), post_size=(10, 20))
    pre_ids, post_ids = conn1.require('pre_ids', 'post_ids')

    conn2 = bp.connect.FixedProb(prob=0.1, seed=123)
    conn2(pre_size=(10, 20), post_size=(10, 20))
    mat = conn2.require('mat')
    pre_ids2, post_ids2 = bp.math.where(mat)


def test_random_fix_pre1():
  for bk in ['numpy', 'jax']:
    bp.math.use_backend(bk)

    for num in [0.4, 20]:
      conn1 = bp.connect.FixedPreNum(num, seed=1234)(pre_size=(10, 15), post_size=(10, 20))
      mat1 = conn1.require('mat')

      conn2 = bp.connect.FixedPreNum(num, seed=1234)(pre_size=(10, 15), post_size=(10, 20))
      mat2 = conn2.require('mat')

      assert bp.math.array_equal(mat1, mat2)


def test_random_fix_pre2():
  for bk in ['numpy', 'jax']:
    bp.math.use_backend(bk)

    for num in [0.5, 3]:
      conn1 = bp.connect.FixedPreNum(num, seed=1234)(pre_size=5, post_size=4)
      mat1 = conn1.require('mat')
      print(mat1)


def test_random_fix_pre3():
  bp.math.use_backend('numpy')
  conn1 = bp.connect.FixedPreNum(num=6, seed=1234)(pre_size=3, post_size=4)
  with pytest.raises(AssertionError):
    conn1.require('mat')


def test_random_fix_post1():
  for bk in ['numpy', 'jax']:
    bp.math.use_backend(bk)

    for num in [0.4, 20]:
      conn1 = bp.connect.FixedPostNum(num, seed=1234)(pre_size=(10, 15), post_size=(10, 20))
      mat1 = conn1.require('mat')

      conn2 = bp.connect.FixedPostNum(num, seed=1234)(pre_size=(10, 15), post_size=(10, 20))
      mat2 = conn2.require('mat')

      assert bp.math.array_equal(mat1, mat2)


def test_random_fix_post2():
  for bk in ['numpy', 'jax']:
    bp.math.use_backend(bk)

    for num in [0.5, 3]:
      conn1 = bp.connect.FixedPostNum(num, seed=1234)(pre_size=5, post_size=4)
      mat1 = conn1.require('mat')
      print(mat1)


def test_random_fix_post3():
  bp.math.use_backend('numpy')
  conn1 = bp.connect.FixedPostNum(num=6, seed=1234)(pre_size=3, post_size=4)
  with pytest.raises(AssertionError):
    conn1.require('mat')
