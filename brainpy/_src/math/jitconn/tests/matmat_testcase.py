# -*- coding: utf-8 -*-

import brainpy.math as bm
import jax
import jax.numpy as jnp
from absl.testing import parameterized

import brainpylib as bl

shapes = [(100, 200),
          (200, 200),
          (10, 1000),
          (2, 1000),
          (1000, 10),
          (1000, 2)]


class Test_matmat_prob_conn(parameterized.TestCase):
  def __init__(self, *args, platform, **kwargs):
    super(Test_matmat_prob_conn, self).__init__(*args, **kwargs)
    bm.set_platform(platform)
    print()

  @parameterized.named_parameters(
    dict(testcase_name=(f'shape = {shape}, '
                        f'm={m}, '
                        f'prob={prob}, '
                        f'w_low = {w_low}, '
                        f'w_high = {w_high}'
                        f'x64 = {x64}'),
         shape=shape,
         prob=prob,
         w_low=w_low,
         w_high=w_high,
         x64=x64,
         m=m,
         seed=1234
         )
    for x64 in [True, False]
    for shape in shapes
    for prob in [0.01, 0.05, 0.1, 0.4]
    for w_low, w_high in [(-1., 0.), (0., 1.), (-1., 1.)]
    for m in [5, 8, 15, 33]
  )
  def test_uniform(self, shape, prob, w_low, w_high, m, seed=None, x64=False):
    print(f'test_uniform: '
          f'shape = {shape}, '
          f'm = {m}, '
          f'prob={prob}, '
          f'w_low = {w_low}, '
          f'w_high = {w_high}, '
          f'x64 = {x64}')

    if x64:
      bm.enable_x64()
    rng = bm.random.RandomState()
    matrix = bm.as_jax(rng.random((m, shape[0])))

    r1 = bl.jitconn_ops.matmat_prob_conn_uniform_weight(matrix,
                                                        w_low=w_low,
                                                        w_high=w_high,
                                                        conn_prob=prob,
                                                        shape=shape,
                                                        seed=seed,
                                                        version='v1')
    r2 = bl.jitconn_ops.matmat_prob_conn_uniform_weight(matrix,
                                                        w_low=w_low,
                                                        w_high=w_high,
                                                        conn_prob=prob,
                                                        shape=shape,
                                                        seed=seed,
                                                        version='v1')
    self.assertTrue(jnp.allclose(r1, r2))

    f = jax.vmap(lambda a: bl.jitconn_ops.matvec_prob_conn_uniform_weight(
      a, w_low=w_low, w_high=w_high, conn_prob=prob, shape=shape, seed=seed, transpose=True))
    r3 = f(matrix)
    self.assertTrue(jnp.allclose(r1, r3))

    if x64:
      bm.disable_x64()
    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    dict(
      testcase_name=(f'test_normal, shape = {shape}, '
                     f'm={m}, '
                     f'prob={prob}, '
                     f'w_mu = {w_mu}, '
                     f'w_sigma = {w_sigma},'
                     f'x64={x64}'),
      shape=shape,
      prob=prob,
      w_mu=w_mu,
      w_sigma=w_sigma,
      seed=1234,
      m=m,
    )
    for x64 in [True, False]
    for shape in shapes
    for prob in [0.01, 0.05, 0.1, 0.2]
    for w_mu, w_sigma in [(-1., 1.), (0., 0.1), (0., 0.5)]
    for m in [5, 8, 15, 33]
  )
  def test_normal(self, shape, prob, w_mu, w_sigma, m, seed=None, x64=False):
    print(f'_test_normal: '
          f'shape = {shape}, '
          f'm = {m}, '
          f'prob={prob}, '
          f'w_mu = {w_mu}, '
          f'w_sigma = {w_sigma}')

    if x64:
      bm.enable_x64()

    rng = bm.random.RandomState()
    matrix = bm.as_jax(rng.random((m, shape[0])))

    r1 = bl.jitconn_ops.matmat_prob_conn_normal_weight(matrix,
                                                       w_mu=w_mu,
                                                       w_sigma=w_sigma,
                                                       conn_prob=prob,
                                                       shape=shape,
                                                       seed=seed)
    r2 = bl.jitconn_ops.matmat_prob_conn_normal_weight(matrix,
                                                       w_mu=w_mu,
                                                       w_sigma=w_sigma,
                                                       conn_prob=prob,
                                                       shape=shape,
                                                       seed=seed)
    self.assertTrue(jnp.allclose(r1, r2))

    f = jax.vmap(
      lambda a: bl.jitconn_ops.matvec_prob_conn_normal_weight(
        a, w_mu=w_mu, w_sigma=w_sigma, conn_prob=prob, shape=shape, seed=seed, transpose=True)
    )
    r3 = f(matrix)
    self.assertTrue(jnp.allclose(r1, r3))

    if x64:
      bm.disable_x64()
    bm.clear_buffer_memory()
