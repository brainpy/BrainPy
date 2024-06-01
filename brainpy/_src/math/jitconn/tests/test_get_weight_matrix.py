# -*- coding: utf-8 -*-
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

import brainpy.math as bm
from brainpy._src.dependency_check import import_taichi

if import_taichi(error_if_not_found=False) is None:
  pytest.skip('no taichi', allow_module_level=True)

import platform

force_test = False  # turn on to force test on windows locally
# if platform.system() == 'Windows' and not force_test:
#   pytest.skip('skip windows', allow_module_level=True)

shapes = [(100, 200), (1000, 10)]

SEED = 1234


class TestGetConnectMatrix(parameterized.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(TestGetConnectMatrix, self).__init__(*args, **kwargs)
    bm.set_platform(platform)
    print()

  @parameterized.product(
    transpose=[True, False],
    outdim_parallel=[True, False],
    shape=shapes,
    prob=[0.1],
  )
  def test_get_conn_matrix(self, transpose, outdim_parallel, shape, prob):
    homo_data = 1.
    print(
      f'test_get_connect_matrix: transpose={transpose}, outdim_parallel={outdim_parallel}, shape={shape}, prob={prob}')
    conn = bm.jitconn.get_homo_weight_matrix(homo_data, prob, SEED, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)
    shape = (shape[1], shape[0]) if transpose else shape
    print(conn.shape)
    assert conn.shape == shape
    # assert conn.dtype == jnp.float_
    # sum all true values
    print(
      f'jnp.sum(conn): {jnp.sum(conn)}, jnp.round(prob * shape[0] * shape[1]): {jnp.round(prob * shape[0] * shape[1])}')

    # compare with jitconn op

    rng = bm.random.RandomState()
    vector = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

    r1 = bm.jitconn.mv_prob_homo(vector,
                                 homo_data,
                                 conn_prob=prob,
                                 shape=shape,
                                 seed=SEED,
                                 outdim_parallel=outdim_parallel,
                                 transpose=transpose)

    r2 = vector @ conn if transpose else conn @ vector
    self.assertTrue(jnp.allclose(r1, r2, atol=1e-6))

    bm.clear_buffer_memory()


class TestGetUniformWeightMatrix(parameterized.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(TestGetUniformWeightMatrix, self).__init__(*args, **kwargs)
    bm.set_platform(platform)
    print()

  @parameterized.product(
    transpose=[True, False],
    outdim_parallel=[True, False],
    shape=shapes,
    prob=[0.1],
    w_low=[0.1],
    w_high=[0.9],
  )
  def test_get_uniform_weight_matrix(self, transpose, outdim_parallel, shape, prob, w_low, w_high):
    print(
      f'test_get_uniform_weight_matrix: transpose={transpose}, outdim_parallel={outdim_parallel}, shape={shape}, prob={prob}, w_low={w_low}, w_high={w_high}')
    weight = bm.jitconn.get_uniform_weight_matrix(w_low, w_high, prob, shape=shape, transpose=transpose,
                                                  outdim_parallel=outdim_parallel)
    shape = (shape[1], shape[0]) if transpose else shape
    assert weight.shape == shape
    assert weight.dtype == jnp.float32

    weight_true = weight > 0.

    print(
      f'jnp.sum(conn): {jnp.sum(weight_true)}, jnp.round(prob * shape[0] * shape[1]): {jnp.round(prob * shape[0] * shape[1])}')

    # CANNOT BE TESTED IN THIS WAY, BECAUSE UNIFORM JITCONN OP HAS BEEN OPTIMIZED
    # compare with jitconn op

    # rng = bm.random.RandomState()
    # events = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))
    #
    # r1 = bm.jitconn.mv_prob_uniform(events,
    #                                 w_low=w_low,
    #                                 w_high=w_high,
    #                                 conn_prob=prob,
    #                                 shape=shape,
    #                                 seed=SEED,
    #                                 outdim_parallel=outdim_parallel,
    #                                 transpose=transpose)
    #
    # r2 = events @ weight if transpose else weight @ events
    # print(f'r1: {r1}\n r2: {r2}')
    # self.assertTrue(jnp.allclose(r1, r2, atol=1e-6))

    bm.clear_buffer_memory()


class TestGetNormalWeightMatrix(parameterized.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(TestGetNormalWeightMatrix, self).__init__(*args, **kwargs)
    bm.set_platform(platform)
    print()

  @parameterized.product(
    transpose=[True, False],
    outdim_parallel=[True, False],
    shape=shapes,
    prob=[0.1],
    w_mu=[0.0],
    w_sigma=[1.0],
  )
  def test_get_normal_weight_matrix(self, transpose, outdim_parallel, shape, prob, w_mu, w_sigma):
    print(
      f'test_get_normal_weight_matrix: transpose={transpose}, outdim_parallel={outdim_parallel}, shape={shape}, prob={prob}, w_mu={w_mu}, w_sigma={w_sigma}')
    weight = bm.jitconn.get_normal_weight_matrix(w_mu, w_sigma, prob, shape=shape, transpose=transpose,
                                                 outdim_parallel=outdim_parallel)
    shape = (shape[1], shape[0]) if transpose else shape
    assert weight.shape == shape
    assert weight.dtype == jnp.float32

    weight_true = weight > 0.

    print(
      f'jnnp.sum(conn): {jnp.sum(weight_true)}, jnp.round(prob * shape[0] * shape[1]): {jnp.round(prob * shape[0] * shape[1])}')

    # CANNOT BE TESTED IN THIS WAY, BECAUSE UNIFORM JITCONN OP HAS BEEN OPTIMIZED
    # compare with jitconn op

    # rng = bm.random.RandomState()
    # vector = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))
    #
    # r1 = bm.jitconn.mv_prob_normal(vector,
    #                                w_mu=w_mu,
    #                                w_sigma=w_sigma,
    #                                conn_prob=prob,
    #                                shape=shape,
    #                                seed=SEED,
    #                                outdim_parallel=outdim_parallel,
    #                                transpose=transpose)
    #
    # r2 = vector @ weight if transpose else weight @ vector
    # print(f'r1: {r1}\n r2: {r2}')
    # self.assertTrue(jnp.allclose(r1, r2, atol=1e-6))

    bm.clear_buffer_memory()
