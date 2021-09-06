# -*- coding: utf-8 -*-

import brainpy as bp


def test_random_prob_numpy():
  bp.math.use_backend('numpy')

  conn1 = bp.connect.FixedProb(prob=0.1, method='iter', seed=123)
  conn1(pre_size=(10, 20), post_size=(10, 20))

  conn2 = bp.connect.FixedProb(prob=0.1, method='matrix', seed=123)
  conn2(pre_size=(10, 20), post_size=(10, 20))

  assert (conn2.pre_ids == conn1.pre_ids).all()
  assert (conn2.post_ids == conn1.post_ids).all()


def test_random_prob_jax():
  bp.math.use_backend('jax')

  conn1 = bp.connect.FixedProb(prob=0.1, method='iter', seed=123)
  conn1(pre_size=(10, 20), post_size=(10, 20))

  conn1 = bp.connect.FixedProb(prob=0.1, method='matrix', seed=123)
  conn1(pre_size=(10, 20), post_size=(10, 20))


def test_random_fix_pre_jax():
  bp.math.use_backend('jax')

  for num in [0.4, 20]:
    conn1 = bp.connect.FixedPreNum(num, method='iter', seed=1234)
    conn1(pre_size=(10, 15), post_size=(10, 20))

    conn2 = bp.connect.FixedPreNum(num, method='matrix', seed=1234)
    conn2(pre_size=(10, 15), post_size=(10, 20))


def test_random_fix_pre_numpy():
  bp.math.use_backend('numpy')

  for num in [0.4, 20]:
    conn1 = bp.connect.FixedPreNum(num, method='iter', seed=1234)
    conn1(pre_size=(10, 15), post_size=(10, 20))

    conn2 = bp.connect.FixedPreNum(num, method='matrix', seed=1234)
    conn2(pre_size=(10, 15), post_size=(10, 20))

    assert (conn2.pre_ids == conn1.pre_ids).all()
    assert (conn2.post_ids == conn1.post_ids).all()


def test_random_fix_post_jax():
  bp.math.use_backend('jax')

  for num in [0.4, 20]:
    conn1 = bp.connect.FixedPostNum(num, method='iter', seed=1234)
    conn1(pre_size=(10, 15), post_size=(10, 20))

    conn2 = bp.connect.FixedPostNum(num, method='matrix', seed=1234)
    conn2(pre_size=(10, 15), post_size=(10, 20))


def test_random_fix_post_numpy():
  bp.math.use_backend('numpy')

  for num in [0.4, 20]:
    conn1 = bp.connect.FixedPostNum(num, method='iter', seed=1234)
    conn1(pre_size=(10, 15), post_size=(10, 20))

    conn2 = bp.connect.FixedPostNum(num, method='matrix', seed=1234)
    conn2(pre_size=(10, 15), post_size=(10, 20))

    # assert (bp.math.sort(conn2.pre_ids) == bp.math.sort(conn1.pre_ids)).all()
    assert (conn2.pre_ids == conn1.pre_ids).all()
    assert (conn2.post_ids == conn1.post_ids).all()





