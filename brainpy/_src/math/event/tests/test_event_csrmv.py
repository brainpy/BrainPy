# -*- coding: utf-8 -*-


from functools import partial

import jax
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm

seed = 1234


def sum_op(op):
  def func(*args, **kwargs):
    r = op(*args, **kwargs)
    return r.sum()

  return func

taichi_csr_matvec = bm.event.csrmv

class Test_event_csr_matvec_taichi(parameterized.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(Test_event_csr_matvec_taichi, self).__init__(*args, **kwargs)

    print()
    bm.set_platform(platform)

  @parameterized.product(
    transpose=[True, False],
    shape=[(100, 200),
           (200, 200),
           (200, 100),
           (10, 1000)],
    homo_data=[-1., 0., 1.],
  )
  def test_homo(self, transpose, shape, homo_data):
    print(f'test_homo: shape = {shape}, transpose = {transpose}, homo_data = {homo_data}')
    rng = bm.random.RandomState(seed=seed)
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    events = rng.random(shape[0] if transpose else shape[1]) < 0.1
    heter_data = bm.ones(indices.shape) * homo_data

    dense = bm.sparse.csr_to_dense(heter_data, indices, indptr, shape=shape)
    r1 = (events @ dense) if transpose else (dense @ events)
    r2 = taichi_csr_matvec(homo_data, indices, indptr, events, shape=shape, transpose=transpose)

    assert (bm.allclose(r1, r2))

    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(100, 200),
           (200, 200),
           (200, 100),
           (10, 1000)],
    homo_data=[-1., 0., 1.],
  )
  def test_homo_vmap(self, shape, transpose, homo_data):
    print(f'test_homo_vamp: shape = {shape}, transpose = {transpose}, homo_data = {homo_data}')

    rng = bm.random.RandomState(seed=seed)
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')

    # vmap 'data'
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
    f1 = jax.vmap(partial(bm.sparse.csrmv, indices=indices, indptr=indptr, vector=events,
                          shape=shape, transpose=transpose))
    f2 = jax.vmap(partial(taichi_csr_matvec, indices=indices, indptr=indptr, events=events,
                          shape=shape, transpose=transpose))
    vmap_data = bm.as_jax([homo_data] * 10)
    self.assertTrue(bm.allclose(f1(vmap_data), f2(vmap_data)))

    # vmap 'events'
    f3 = jax.vmap(partial(bm.sparse.csrmv, homo_data, indices, indptr,
                          shape=shape, transpose=transpose))
    f4 = jax.vmap(partial(taichi_csr_matvec, homo_data, indices, indptr,
                          shape=shape, transpose=transpose))
    vmap_data = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.1
    self.assertTrue(bm.allclose(f3(vmap_data), f4(vmap_data)))

    # vmap 'data' and 'events'
    f5 = jax.vmap(lambda dd, ee: bm.sparse.csrmv(dd, indices, indptr, ee, shape=shape, transpose=transpose))
    f6 = jax.vmap(lambda dd, ee: taichi_csr_matvec(dd, indices, indptr, ee, shape=shape, transpose=transpose))

    vmap_data1 = bm.as_jax([homo_data] * 10)
    vmap_data2 = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.2
    self.assertTrue(bm.allclose(f5(vmap_data1, vmap_data2),
                                f6(vmap_data1, vmap_data2)))

    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(100, 200),
           (200, 200),
           (200, 100),
           (10, 1000)],
    homo_data=[-1., 0., 1.],
  )
  def test_homo_grad(self, shape, transpose, homo_data):
    print(f'test_homo_grad: shape = {shape}, transpose = {transpose}, homo_data = {homo_data}')

    rng = bm.random.RandomState(seed=seed)
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
    dense_conn = bm.sparse.csr_to_dense(bm.ones(indices.shape).value, indices, indptr, shape=shape)

    # grad 'data'
    r1 = jax.grad(sum_op(bm.sparse.csrmv))(
      homo_data, indices, indptr, events, shape=shape, transpose=transpose)
    r2 = jax.grad(sum_op(taichi_csr_matvec))(
      homo_data, indices, indptr, events, shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r1, r2))

    # grad 'events'
    r3 = jax.grad(sum_op(bm.sparse.csrmv), argnums=3)(
      homo_data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    r4 = jax.grad(sum_op(taichi_csr_matvec), argnums=3)(
      homo_data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r3, r4))

    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(100, 200),
           (200, 200),
           (200, 100),
           (10, 1000), ]
  )
  def test_heter(self, shape, transpose):
    print(f'test_heter: shape = {shape}, transpose = {transpose}')
    rng = bm.random.RandomState(seed=seed)
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
    heter_data = bm.as_jax(rng.random(indices.shape))

    r1 = bm.sparse.csrmv(heter_data, indices, indptr, events,
                        shape=shape, transpose=transpose)
    r2 = taichi_csr_matvec(heter_data, indices, indptr, events,
                               shape=shape, transpose=transpose)

    assert (bm.allclose(r1, r2))

    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(100, 200),
           (200, 200),
           (200, 100),
           (10, 1000)]
  )
  def test_heter_vmap(self, shape, transpose):
    print(f'test_heter_vamp: shape = {shape}, transpose = {transpose}')

    rng = bm.random.RandomState(seed=seed)
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)

    # vmap 'data'
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
    f1 = jax.vmap(partial(bm.sparse.csrmv, indices=indices, indptr=indptr,  vector=events,
                          shape=shape, transpose=transpose))
    f2 = jax.vmap(partial(taichi_csr_matvec, indices=indices, indptr=indptr, events=events,
                          shape=shape, transpose=transpose))
    vmap_data = bm.as_jax(rng.random((10, indices.shape[0])))
    self.assertTrue(bm.allclose(f1(vmap_data), f2(vmap_data)))

    # vmap 'events'
    data = bm.as_jax(rng.random(indices.shape))
    f3 = jax.vmap(partial(bm.sparse.csrmv, data, indices, indptr,
                          shape=shape, transpose=transpose))
    f4 = jax.vmap(partial(taichi_csr_matvec, data, indices, indptr,
                          shape=shape, transpose=transpose))
    vmap_data = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.1
    self.assertTrue(bm.allclose(f3(vmap_data), f4(vmap_data)))

    # vmap 'data' and 'events'
    f5 = jax.vmap(lambda dd, ee: bm.sparse.csrmv(dd, indices, indptr, ee,
                                                shape=shape, transpose=transpose))
    f6 = jax.vmap(lambda dd, ee: taichi_csr_matvec(dd, indices, indptr, ee,
                                                       shape=shape, transpose=transpose))
    vmap_data1 = bm.as_jax(rng.random((10, indices.shape[0])))
    vmap_data2 = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.2
    self.assertTrue(bm.allclose(f5(vmap_data1, vmap_data2),
                                f6(vmap_data1, vmap_data2)))

    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(100, 200),
           (200, 200),
           (200, 100),
           (10, 1000)]
  )
  def test_heter_grad(self, shape, transpose):
    print(f'test_heter_grad: shape = {shape}, transpose = {transpose}')

    rng = bm.random.RandomState(seed=seed)
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    events = rng.random(shape[0] if transpose else shape[1]) < 0.1
    events = bm.as_jax(events)
    dense_conn = bm.sparse.csr_to_dense(bm.ones(indices.shape).value, indices, indptr, shape=shape)

    # grad 'data'
    data = bm.as_jax(rng.random(indices.shape))
    r1 = jax.grad(sum_op(bm.sparse.csrmv))(
      data, indices, indptr, events, shape=shape, transpose=transpose)
    r2 = jax.grad(sum_op(taichi_csr_matvec))(
      data, indices, indptr, events, shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r1, r2))

    # grad 'events'
    r3 = jax.grad(sum_op(bm.sparse.csrmv), argnums=3)(
      data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    r4 = jax.grad(sum_op(taichi_csr_matvec), argnums=3)(
      data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r3, r4))

    r5 = jax.grad(sum_op(bm.sparse.csrmv), argnums=(0, 3))(
      data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    r6 = jax.grad(sum_op(taichi_csr_matvec), argnums=(0, 3))(
      data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r5[0], r6[0]))
    self.assertTrue(bm.allclose(r5[1], r6[1]))

    bm.clear_buffer_memory()
