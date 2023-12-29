# -*- coding: utf-8 -*-

import sys

import jax
import jax.numpy as jnp
import pytest
from absl.testing import parameterized

import brainpy.math as bm

is_manual_test = False
if sys.platform.startswith('darwin') and not is_manual_test:
  pytest.skip('brainpy.math package may need manual tests.', allow_module_level=True)

shapes = [(100, 200), (10, 1000), (2, 1000), (1000, 10), (1000, 2)]
shapes = [(100, 200), (2, 1000), (1000, 2)]


# def sum_op(op):
#     def func(*args, **kwargs):
#         r = op(*args, **kwargs)
#         return r.sum()

#     return func


# def sum_op2(op):
#     def func(*args, **kwargs):
#         r = op(*args, **kwargs)[0]
#         return r.sum()

#     return func

# def test_homo(shape, transpose, outdim_parallel, prob, homo_data, seed=None):
#     print(f'test_homo: '
#           f'shape = {shape}, '
#           f'transpose = {transpose}, '
#           f'outdim_parallel = {outdim_parallel}, '
#           f'prob={prob}, '
#           f'homo_data = {homo_data}')

#     rng = bm.random.RandomState()
#     vector = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

#     r1 = bm.jitconn.mv_prob_homo_taichi(vector,
#                                  homo_data,
#                                  conn_prob=prob,
#                                  shape=shape,
#                                  seed=seed,
#                                  outdim_parallel=outdim_parallel,
#                                  transpose=transpose)

#     r2 = bm.jitconn.mv_prob_homo_taichi(vector,
#                                  homo_data,
#                                  conn_prob=prob,
#                                  shape=shape,
#                                  seed=seed,
#                                  outdim_parallel=outdim_parallel,
#                                  transpose=transpose)
#     assert (jnp.allclose(r1, r2))

#     r2 = bm.jitconn.mv_prob_homo_taichi(vector,
#                                  homo_data,
#                                  conn_prob=prob,
#                                  shape=(shape[1], shape[0]),
#                                  seed=seed,
#                                  outdim_parallel=outdim_parallel,
#                                  transpose=not transpose)
#     assert (jnp.allclose(r1, r2))

#     bm.clear_buffer_memory()

# def test_homo_vmap(shape, transpose, outdim_parallel, prob, seed=None):
#     print(f'test_homo_vmap: '
#           f'shape = {shape}, '
#           f'transpose = {transpose}, '
#           f'outdim_parallel = {outdim_parallel}, '
#           f'prob={prob}')

#     rng = bm.random.RandomState()
#     events = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1])))
#     weights = bm.as_jax(rng.random(10))

#     f1 = jax.vmap(
#       lambda event, data: bm.jitconn.mv_prob_homo_taichi(
#         event, data,
#         conn_prob=prob, shape=shape, seed=seed,
#         outdim_parallel=outdim_parallel, transpose=transpose
#       )[0]
#     )
#     r1 = f1(events, weights)
#     r2 = f1(events, weights)
#     assert (jnp.allclose(r1, r2))

#     bm.clear_buffer_memory()

# def test_uniform(shape, transpose, outdim_parallel, prob, w_low, w_high, seed=None):
#     print(f'test_uniform: '
#           f'shape = {shape}, '
#           f'transpose = {transpose}, '
#           f'outdim_parallel = {outdim_parallel}, '
#           f'prob={prob}, '
#           f'w_low = {w_low}, '
#           f'w_high = {w_high}, ')

#     rng = bm.random.RandomState()
#     events = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

#     r1 = bm.jitconn.mv_prob_uniform_taichi(events,
#                                     w_low=w_low,
#                                     w_high=w_high,
#                                     conn_prob=prob,
#                                     shape=shape,
#                                     seed=seed,
#                                     outdim_parallel=outdim_parallel,
#                                     transpose=transpose)

#     r2 = bm.jitconn.mv_prob_uniform_taichi(events,
#                                     w_low=w_low,
#                                     w_high=w_high,
#                                     conn_prob=prob,
#                                     shape=shape,
#                                     seed=seed,
#                                     outdim_parallel=outdim_parallel,
#                                     transpose=transpose)
#     c = jnp.allclose(r1, r2)
#     if not c:
#       print(r1, r2)
#     assert (c)

#     r2 = bm.jitconn.mv_prob_uniform_taichi(events,
#                                     w_low=w_low,
#                                     w_high=w_high,
#                                     conn_prob=prob,
#                                     shape=(shape[1], shape[0]),
#                                     seed=seed,
#                                     outdim_parallel=outdim_parallel,
#                                     transpose=not transpose)
#     c = jnp.allclose(r1, r2)
#     if not c:
#       print(r1, r2)
#     assert (c)

#     bm.clear_buffer_memory()

# test_homo(shape=(100, 200), transpose=True, outdim_parallel=True, prob=0.1, homo_data=1., seed=1234)
# test_homo_vmap(shape=(100, 200), transpose=True, outdim_parallel=True, prob=0.1, seed=1234)

# test_uniform(shape=(100, 200), transpose=True, outdim_parallel=False, prob=0.1, w_low=-1., w_high=0., seed=1234)

# def test_homo_grad(shape, transpose, outdim_parallel, prob, seed=None):
#     print(f'_test_homo_grad: '
#           f'shape = {shape}, '
#           f'transpose = {transpose}, '
#           f'outdim_parallel = {outdim_parallel}, '
#           f'prob={prob}')

#     rng = bm.random.RandomState()
#     events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.5
#     events = events.astype(float)

#     f1 = jax.grad(
#       lambda event, data: bm.jitconn.mv_prob_homo_taichi(
#         event, data,
#         conn_prob=prob,
#         shape=shape,
#         seed=seed,
#         outdim_parallel=outdim_parallel,
#         transpose=transpose
#       )[0].sum(),
#       argnums=0
#     )
#     r1 = f1(events, 1.)
#     r2 = f1(events, 2.)

#     print(r1 *2 - r2)
#     assert (jnp.allclose(r1 * 2., r2))

#     bm.clear_buffer_memory()


# def test_normal_grad(shape, transpose, outdim_parallel, prob, seed=None):
#     print(f'_test_normal_grad: '
#           f'shape = {shape}, '
#           f'transpose = {transpose}, '
#           f'outdim_parallel = {outdim_parallel}, '
#           f'prob={prob}')

#     rng = bm.random.RandomState()
#     events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
#     events = events.astype(float)

#     f1 = jax.grad(
#       lambda e, w_sigma: bm.jitconn.mv_prob_normal_taichi(
#         e,
#         w_mu=0.,
#         w_sigma=w_sigma,
#         conn_prob=prob,
#         shape=shape,
#         seed=seed,
#         outdim_parallel=outdim_parallel,
#         transpose=transpose
#       )[0].sum()
#     )
#     r1 = f1(events, 1.)
#     r2 = f1(events, 2.)
#     print(r1 *2 - r2)
#     assert (bm.allclose(r1 * 2., r2))

#     bm.clear_buffer_memory()

# def test_uniform_grad(shape, transpose, outdim_parallel, prob, seed=None):
#     print(f'_test_uniform_grad: '
#           f'shape = {shape}, '
#           f'transpose = {transpose}, '
#           f'outdim_parallel = {outdim_parallel}, '
#           f'prob={prob}')


#     rng = bm.random.RandomState()
#     events = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

#     f1 = jax.grad(
#       lambda e, w_low, w_high: bm.jitconn.mv_prob_uniform_taichi(
#         e,
#         w_low=w_low,
#         w_high=w_high,
#         conn_prob=prob,
#         shape=shape,
#         seed=seed,
#         outdim_parallel=outdim_parallel,
#         transpose=transpose
#       )[0].sum()
#     )

#     r1 = f1(events, 0., 1.)
#     r2 = f1(events, 0., 2.)
#     print(r1 *2 - r2)
#     assert (bm.allclose(r1 * 2., r2))

#     bm.clear_buffer_memory()

# test_homo_grad(shape=(100, 200), transpose=True, outdim_parallel=True, prob=0.1, seed=1234)
# test_normal_grad(shape=(100, 200), transpose=True, outdim_parallel=True, prob=0.1, seed=1234)
# test_uniform_grad(shape=(100, 200), transpose=True, outdim_parallel=False, prob=0.1, seed=1234)


class Test_matvec_prob_conn(parameterized.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(Test_matvec_prob_conn, self).__init__(*args, **kwargs)
    bm.set_platform(platform)
    print()

  @parameterized.named_parameters(
    dict(testcase_name=(f'test_homo, shape = {shape}, '
                        f'transpose = {transpose}, '
                        f'outdim_parallel = {outdim_parallel}, '
                        f'prob={prob}, '
                        f'homo_data = {homo_data}, '
                        f'x64 = {x64}'),
         shape=shape,
         transpose=transpose,
         outdim_parallel=outdim_parallel,
         prob=prob,
         homo_data=homo_data,
         seed=1234)
    for x64 in [True, False]
    for transpose in [True, False]
    for outdim_parallel in [True, False]
    for shape in shapes
    for prob in [0.01, 0.1]
    for homo_data in [-1., 1.]
  )
  def test_homo(self, shape, transpose, outdim_parallel, prob, homo_data, seed=None, x64=False):
    print(f'test_homo: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'outdim_parallel = {outdim_parallel}, '
          f'prob={prob}, '
          f'homo_data = {homo_data}')

    if x64:
      bm.enable_x64()

    rng = bm.random.RandomState()
    vector = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

    r1 = bm.jitconn.mv_prob_homo_taichi(vector,
                                        homo_data,
                                        conn_prob=prob,
                                        shape=shape,
                                        seed=seed,
                                        outdim_parallel=outdim_parallel,
                                        transpose=transpose)

    r2 = bm.jitconn.mv_prob_homo_taichi(vector,
                                        homo_data,
                                        conn_prob=prob,
                                        shape=shape,
                                        seed=seed,
                                        outdim_parallel=outdim_parallel,
                                        transpose=transpose)
    self.assertTrue(jnp.allclose(r1, r2))

    r2 = bm.jitconn.mv_prob_homo_taichi(vector,
                                        homo_data,
                                        conn_prob=prob,
                                        shape=(shape[1], shape[0]),
                                        seed=seed,
                                        outdim_parallel=outdim_parallel,
                                        transpose=not transpose)
    self.assertTrue(jnp.allclose(r1, r2))

    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    dict(testcase_name=(f'test_homo_vmap, shape = {shape}, '
                        f'transpose = {transpose}, '
                        f'outdim_parallel = {outdim_parallel}, '
                        f'prob={prob}, x64={x64}'),
         shape=shape,
         transpose=transpose,
         outdim_parallel=outdim_parallel,
         prob=prob,
         seed=1234,
         x64=x64)
    for transpose in [True, False]
    for x64 in [True, False]
    for outdim_parallel in [True, False]
    for shape in shapes
    for prob in [0.01, 0.1]
  )
  def test_homo_vmap(self, shape, transpose, outdim_parallel, prob, seed=None, x64=False):
    print(f'test_homo_vmap: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'outdim_parallel = {outdim_parallel}, '
          f'prob={prob}')

    if x64:
      bm.enable_x64()

    rng = bm.random.RandomState()
    events = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1])))
    weights = bm.as_jax(rng.random(10))

    f1 = jax.vmap(
      lambda event, data: bm.jitconn.mv_prob_homo_taichi(
        event, data,
        conn_prob=prob, shape=shape, seed=seed,
        outdim_parallel=outdim_parallel, transpose=transpose
      )[0]
    )
    r1 = f1(events, weights)
    r2 = f1(events, weights)
    self.assertTrue(jnp.allclose(r1, r2))

    if x64:
      bm.disable_x64()
    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    dict(testcase_name=(f'test_homo_grad, shape = {shape}, '
                        f'transpose = {transpose}, '
                        f'outdim_parallel = {outdim_parallel}, '
                        f'prob={prob}, x64={x64}'),
         shape=shape,
         transpose=transpose,
         outdim_parallel=outdim_parallel,
         prob=prob,
         seed=1234,
         x64=x64)
    for transpose in [True, False]
    for x64 in [True, False]
    for outdim_parallel in [True, False]
    for shape in shapes
    for prob in [0.01, 0.1]
  )
  def test_homo_grad(self, shape, transpose, outdim_parallel, prob, seed=None, x64=False):
    print(f'_test_homo_grad: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'outdim_parallel = {outdim_parallel}, '
          f'prob={prob}')

    if x64:
      bm.enable_x64()
    rng = bm.random.RandomState()
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.5
    events = events.astype(float)

    f1 = jax.grad(
      lambda event, data: bm.jitconn.mv_prob_homo_taichi(
        event, data,
        conn_prob=prob,
        shape=shape,
        seed=seed,
        outdim_parallel=outdim_parallel,
        transpose=transpose
      )[0].sum(),
      argnums=0
    )
    r1 = f1(events, 1.)
    r2 = f1(events, 2.)

    self.assertTrue(jnp.allclose(r1 * 2., r2))

    if x64:
      bm.disable_x64()
    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    dict(testcase_name=(f'test_uniform, shape = {shape}, '
                        f'transpose = {transpose}, '
                        f'outdim_parallel = {outdim_parallel}, '
                        f'prob={prob}, '
                        f'w_low = {w_low}, '
                        f'w_high = {w_high}'
                        f'x64 = {x64}'),
         shape=shape,
         transpose=transpose,
         outdim_parallel=outdim_parallel,
         prob=prob,
         w_low=w_low,
         w_high=w_high,
         x64=x64,
         seed=1234)
    for x64 in [True, False]
    for transpose in [True, False]
    for outdim_parallel in [True, False]
    for shape in shapes
    for prob in [0.01, 0.1]
    for w_low, w_high in [(-1., 0.), (0., 1.), (-1., 1.)]
  )
  def test_uniform(self, shape, transpose, outdim_parallel, prob, w_low, w_high, seed=None, x64=False):
    print(f'test_uniform: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'outdim_parallel = {outdim_parallel}, '
          f'prob={prob}, '
          f'w_low = {w_low}, '
          f'w_high = {w_high}, '
          f'x64 = {x64}')

    if x64:
      bm.enable_x64()
    rng = bm.random.RandomState()
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

    r1 = bm.jitconn.mv_prob_uniform_taichi(events,
                                           w_low=w_low,
                                           w_high=w_high,
                                           conn_prob=prob,
                                           shape=shape,
                                           seed=seed,
                                           outdim_parallel=outdim_parallel,
                                           transpose=transpose)

    r2 = bm.jitconn.mv_prob_uniform_taichi(events,
                                           w_low=w_low,
                                           w_high=w_high,
                                           conn_prob=prob,
                                           shape=shape,
                                           seed=seed,
                                           outdim_parallel=outdim_parallel,
                                           transpose=transpose)
    c = jnp.allclose(r1, r2)
    if not c:
      print(r1, r2)
    self.assertTrue(c)

    r2 = bm.jitconn.mv_prob_uniform_taichi(events,
                                           w_low=w_low,
                                           w_high=w_high,
                                           conn_prob=prob,
                                           shape=(shape[1], shape[0]),
                                           seed=seed,
                                           outdim_parallel=outdim_parallel,
                                           transpose=not transpose)
    c = jnp.allclose(r1, r2)
    if not c:
      print(r1, r2)
    self.assertTrue(c)

    if x64:
      bm.disable_x64()
    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    dict(testcase_name=f'test_uniform_vmap, shape = {shape}, '
                       f'transpose = {transpose}, '
                       f'outdim_parallel = {outdim_parallel}, '
                       f'prob={prob}, x64={x64}',
         shape=shape,
         transpose=transpose,
         outdim_parallel=outdim_parallel,
         prob=prob,
         seed=1234,
         x64=x64)
    for transpose in [True, False]
    for x64 in [True, False]
    for outdim_parallel in [True, False]
    for shape in shapes
    for prob in [0.01, 0.1]
  )
  def test_uniform_vmap(self, shape, transpose, outdim_parallel, prob, seed=None, x64=False):
    print(f'test_uniform_vmap: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'outdim_parallel = {outdim_parallel}, '
          f'prob={prob}')

    if x64:
      bm.enable_x64()
    rng = bm.random.RandomState()
    events = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1])))

    f1 = jax.vmap(lambda e: bm.jitconn.mv_prob_uniform_taichi(e,
                                                              w_low=0.,
                                                              w_high=1.,
                                                              conn_prob=prob,
                                                              shape=shape,
                                                              seed=seed,
                                                              outdim_parallel=outdim_parallel,
                                                              transpose=transpose))

    r1 = f1(events)
    r2 = f1(events)
    self.assertTrue(jnp.allclose(r1, r2))

    if x64:
      bm.disable_x64()
    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    dict(testcase_name=(f'test_uniform_grad, shape = {shape}, '
                        f'transpose = {transpose}, '
                        f'outdim_parallel = {outdim_parallel}, '
                        f'prob={prob}, '
                        f'x64={x64}'),
         shape=shape,
         transpose=transpose,
         outdim_parallel=outdim_parallel,
         prob=prob,
         seed=1234,
         x64=x64)
    for x64 in [True, False]
    for transpose in [True, False]
    for outdim_parallel in [True, False]
    for shape in shapes
    for prob in [0.01, 0.1]
  )
  def test_uniform_grad(self, shape, transpose, outdim_parallel, prob, seed=None, x64=False):
    print(f'_test_uniform_grad: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'outdim_parallel = {outdim_parallel}, '
          f'prob={prob}')

    if x64:
      bm.enable_x64()

    rng = bm.random.RandomState()
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

    f1 = jax.grad(
      lambda e, w_low, w_high: bm.jitconn.mv_prob_uniform_taichi(
        e,
        w_low=w_low,
        w_high=w_high,
        conn_prob=prob,
        shape=shape,
        seed=seed,
        outdim_parallel=outdim_parallel,
        transpose=transpose
      )[0].sum()
    )

    r1 = f1(events, 0., 1.)
    r2 = f1(events, 0., 2.)

    self.assertTrue(bm.allclose(r1 * 2., r2))

    if x64:
      bm.disable_x64()
    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    dict(
      testcase_name=(f'test_normal, shape = {shape}, '
                     f'transpose = {transpose}, '
                     f'outdim_parallel = {outdim_parallel}, '
                     f'prob={prob}, '
                     f'w_mu = {w_mu}, '
                     f'w_sigma = {w_sigma},'
                     f'x64={x64}'),
      shape=shape,
      transpose=transpose,
      outdim_parallel=outdim_parallel,
      prob=prob,
      w_mu=w_mu,
      w_sigma=w_sigma,
      seed=1234
    )
    for transpose in [True, False]
    for x64 in [True, False]
    for outdim_parallel in [True, False]
    for shape in shapes
    for prob in [0.01, 0.1]
    for w_mu, w_sigma in [(-1., 1.), (0., 0.1), (0., 0.5)]
  )
  def test_normal(self, shape, transpose, outdim_parallel, prob, w_mu, w_sigma, seed=None, x64=False):
    print(f'_test_normal: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'outdim_parallel = {outdim_parallel}, '
          f'prob={prob}, '
          f'w_mu = {w_mu}, '
          f'w_sigma = {w_sigma}')

    if x64:
      bm.enable_x64()

    rng = bm.random.RandomState()
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

    r1 = bm.jitconn.mv_prob_normal_taichi(events,
                                          w_mu=w_mu,
                                          w_sigma=w_sigma,
                                          conn_prob=prob,
                                          shape=shape,
                                          seed=seed,
                                          outdim_parallel=outdim_parallel,
                                          transpose=transpose)

    r2 = bm.jitconn.mv_prob_normal_taichi(events,
                                          w_mu=w_mu,
                                          w_sigma=w_sigma,
                                          conn_prob=prob,
                                          shape=shape,
                                          seed=seed,
                                          outdim_parallel=outdim_parallel,
                                          transpose=transpose)
    c = jnp.allclose(r1, r2)
    if not c:
      print(r1, r2)
    self.assertTrue(c)

    r2 = bm.jitconn.mv_prob_normal_taichi(events,
                                          w_mu=w_mu,
                                          w_sigma=w_sigma,
                                          conn_prob=prob,
                                          shape=(shape[1], shape[0]),
                                          seed=seed,
                                          outdim_parallel=outdim_parallel,
                                          transpose=not transpose)
    c = jnp.allclose(r1, r2)
    if not c:
      print(r1, r2)
    self.assertTrue(c)

    if x64:
      bm.disable_x64()
    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    dict(testcase_name=f'test_normal_vmap, shape = {shape}, '
                       f'transpose = {transpose}, '
                       f'outdim_parallel = {outdim_parallel}, '
                       f'prob={prob}, '
                       f'x64={x64}',
         shape=shape,
         transpose=transpose,
         outdim_parallel=outdim_parallel,
         prob=prob,
         seed=1234)
    for transpose in [True, False]
    for x64 in [True, False]
    for outdim_parallel in [True, False]
    for shape in shapes
    for prob in [0.01, 0.1]
  )
  def test_normal_vmap(self, shape, transpose, outdim_parallel, prob, seed=None, x64=False):
    print(f'_test_normal_vmap: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'outdim_parallel = {outdim_parallel}, '
          f'prob={prob}')

    if x64:
      bm.enable_x64()

    rng = bm.random.RandomState()
    events = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1])))

    f1 = jax.vmap(lambda e: bm.jitconn.mv_prob_normal_taichi(e,
                                                             w_mu=0.,
                                                             w_sigma=1.,
                                                             conn_prob=prob,
                                                             shape=shape,
                                                             seed=seed,
                                                             outdim_parallel=outdim_parallel,
                                                             transpose=transpose))
    r1 = f1(events)
    r2 = f1(events)
    c = jnp.allclose(r1, r2, atol=1e-6)
    if not c:
      print(r1, r2)
      print(r1 - r2)
    self.assertTrue(c)

    if x64:
      bm.disable_x64()
    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    dict(shape=shape,
         transpose=transpose,
         outdim_parallel=outdim_parallel,
         prob=prob,
         seed=1234,
         x64=x64,
         testcase_name=f'test_normal_grad: '
                       f'shape = {shape}, '
                       f'transpose = {transpose}, '
                       f'outdim_parallel = {outdim_parallel}, '
                       f'prob={prob}, '
                       f'x64={x64}')
    for transpose in [True, False]
    for x64 in [True, False]
    for outdim_parallel in [True, False]
    for shape in shapes
    for prob in [0.01, 0.1]
  )
  def test_normal_grad(self, shape, transpose, outdim_parallel, prob, seed=None, x64=False):
    print(f'_test_normal_grad: '
          f'shape = {shape}, '
          f'transpose = {transpose}, '
          f'outdim_parallel = {outdim_parallel}, '
          f'prob={prob}')

    if x64:
      bm.enable_x64()
    rng = bm.random.RandomState()
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
    events = events.astype(float)

    f1 = jax.grad(
      lambda e, w_sigma: bm.jitconn.mv_prob_normal_taichi(
        e,
        w_mu=0.,
        w_sigma=w_sigma,
        conn_prob=prob,
        shape=shape,
        seed=seed,
        outdim_parallel=outdim_parallel,
        transpose=transpose
      )[0].sum()
    )
    r1 = f1(events, 1.)
    r2 = f1(events, 2.)
    self.assertTrue(bm.allclose(r1 * 2., r2))

    if x64:
      bm.disable_x64()
    bm.clear_buffer_memory()
