# -*- coding: utf-8 -*-


import sys
from functools import partial

import jax
import pytest
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm

# pytestmark = pytest.mark.skip(reason="Skipped due to pytest limitations, manual execution required for testing.")

is_manual_test = False
if sys.platform.startswith('darwin') and not is_manual_test:
  pytest.skip('brainpy.math package may need manual tests.', allow_module_level=True)

# bm.set_platform('cpu')

seed = 1234


def sum_op(op):
  def func(*args, **kwargs):
    r = op(*args, **kwargs)
    return r.sum()

  return func


def sum_op2(op):
  def func(*args, **kwargs):
    r = op(*args, **kwargs)[0]
    return r.sum()

  return func


# ### MANUAL TESTS ###

# transposes = [True, False]
# shapes = [(100, 200),
#          (200, 200),
#          (200, 100),
#          (10, 1000),
#          # (2, 10000),
#          # (1000, 10),
#          # (10000, 2)
#           ]
# homo_datas = [-1., 0., 1.]

# def test_homo(shape, transpose, homo_data):
#   print(f'test_homo: shape = {shape}, transpose = {transpose}, homo_data = {homo_data}')
#   rng = bm.random.RandomState()
#   indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
#   events = rng.random(shape[0] if transpose else shape[1]) < 0.1
#   heter_data = bm.ones(indices.shape) * homo_data

#   r1 = bm.event.csrmv(homo_data, indices, indptr, events, shape=shape, transpose=transpose)
#   r2 = bm.event.csrmv_taichi(homo_data, indices, indptr, events, shape=shape, transpose=transpose)

#   assert (bm.allclose(r1, r2[0]))

#   bm.clear_buffer_memory()


# def test_homo_vmap(shape, transpose, homo_data):
#   print(f'test_homo_vamp: shape = {shape}, transpose = {transpose}, homo_data = {homo_data}')

#   rng = bm.random.RandomState()
#   indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')

#   # vmap 'data'
#   events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
#   f1 = jax.vmap(partial(bm.event.csrmv, indices=indices, indptr=indptr, events=events,
#                         shape=shape, transpose=transpose))
#   f2 = jax.vmap(partial(bm.event.csrmv_taichi, indices=indices, indptr=indptr, events=events,
#                         shape=shape, transpose=transpose))
#   vmap_data = bm.as_jax([homo_data] * 10)
#   assert(bm.allclose(f1(vmap_data), f2(vmap_data)[0]))

#   # vmap 'events'
#   f3 = jax.vmap(partial(bm.event.csrmv, homo_data, indices, indptr,
#                         shape=shape, transpose=transpose))
#   f4 = jax.vmap(partial(bm.event.csrmv_taichi, homo_data, indices, indptr,
#                         shape=shape, transpose=transpose))
#   vmap_data = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.1
#   assert(bm.allclose(f3(vmap_data), f4(vmap_data)[0]))

#   # vmap 'data' and 'events'
#   f5 = jax.vmap(lambda dd, ee: bm.event.csrmv(dd, indices, indptr, ee, shape=shape, transpose=transpose))
#   f6 = jax.vmap(lambda dd, ee: bm.event.csrmv_taichi(dd, indices, indptr, ee, shape=shape, transpose=transpose))

#   vmap_data1 = bm.as_jax([homo_data] * 10)
#   vmap_data2 = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.2
#   assert(bm.allclose(f5(vmap_data1, vmap_data2),
#                                 f6(vmap_data1, vmap_data2)[0]))

#   bm.clear_buffer_memory()


# def test_homo_grad(shape, transpose, homo_data):
#   print(f'test_homo_grad: shape = {shape}, transpose = {transpose}, homo_data = {homo_data}')

#   rng = bm.random.RandomState()
#   indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
#   indices = bm.as_jax(indices)
#   indptr = bm.as_jax(indptr)
#   events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
#   dense_conn = bm.sparse.csr_to_dense(bm.ones(indices.shape).value, indices, indptr, shape=shape)

#   # grad 'data'
#   r1 = jax.grad(sum_op(bm.event.csrmv))(
#     homo_data, indices, indptr, events, shape=shape, transpose=transpose)
#   r2 = jax.grad(sum_op2(bm.event.csrmv_taichi))(
#     homo_data, indices, indptr, events, shape=shape, transpose=transpose)
#   assert(bm.allclose(r1, r2))

#   # grad 'events'
#   r3 = jax.grad(sum_op(bm.event.csrmv), argnums=3)(
#     homo_data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
#   r4 = jax.grad(sum_op2(bm.event.csrmv_taichi), argnums=3)(
#     homo_data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
#   assert(bm.allclose(r3, r4))

#   bm.clear_buffer_memory()


# def test_heter(shape, transpose):
#   print(f'test_heter: shape = {shape}, transpose = {transpose}')
#   rng = bm.random.RandomState()
#   indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
#   indices = bm.as_jax(indices)
#   indptr = bm.as_jax(indptr)
#   events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
#   heter_data = bm.as_jax(rng.random(indices.shape))

#   r1 = bm.event.csrmv(heter_data, indices, indptr, events,
#                           shape=shape, transpose=transpose)
#   r2 = bm.event.csrmv_taichi(heter_data, indices, indptr, events,
#                               shape=shape, transpose=transpose)

#   assert(bm.allclose(r1, r2[0]))

#   bm.clear_buffer_memory()


# def test_heter_vmap(shape, transpose):
#   print(f'test_heter_vamp: shape = {shape}, transpose = {transpose}')

#   rng = bm.random.RandomState()
#   indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
#   indices = bm.as_jax(indices)
#   indptr = bm.as_jax(indptr)

#   # vmap 'data'
#   events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
#   f1 = jax.vmap(partial(bm.event.csrmv, indices=indices, indptr=indptr, events=events,
#                         shape=shape, transpose=transpose))
#   f2 = jax.vmap(partial(bm.event.csrmv_taichi, indices=indices, indptr=indptr, events=events,
#                         shape=shape, transpose=transpose))
#   vmap_data = bm.as_jax(rng.random((10, indices.shape[0])))
#   assert(bm.allclose(f1(vmap_data), f2(vmap_data)[0]))

#   # vmap 'events'
#   data = bm.as_jax(rng.random(indices.shape))
#   f3 = jax.vmap(partial(bm.event.csrmv, data, indices, indptr,
#                         shape=shape, transpose=transpose))
#   f4 = jax.vmap(partial(bm.event.csrmv_taichi, data, indices, indptr,
#                         shape=shape, transpose=transpose))
#   vmap_data = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.1
#   assert(bm.allclose(f3(vmap_data), f4(vmap_data)[0]))

#   # vmap 'data' and 'events'
#   f5 = jax.vmap(lambda dd, ee: bm.event.csrmv(dd, indices, indptr, ee,
#                                               shape=shape, transpose=transpose))
#   f6 = jax.vmap(lambda dd, ee: bm.event.csrmv_taichi(dd, indices, indptr, ee,
#                                               shape=shape, transpose=transpose))
#   vmap_data1 = bm.as_jax(rng.random((10, indices.shape[0])))
#   vmap_data2 = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.2
#   assert(bm.allclose(f5(vmap_data1, vmap_data2),
#                                 f6(vmap_data1, vmap_data2)[0]))

#   bm.clear_buffer_memory()


# def test_heter_grad(shape, transpose):
#   print(f'test_heter_grad: shape = {shape}, transpose = {transpose}')

#   rng = bm.random.RandomState()
#   indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
#   indices = bm.as_jax(indices)
#   indptr = bm.as_jax(indptr)
#   events = rng.random(shape[0] if transpose else shape[1]) < 0.1
#   events = bm.as_jax(events)
#   dense_conn = bm.sparse.csr_to_dense(bm.ones(indices.shape).value, indices, indptr, shape=shape)

#   # grad 'data'
#   data = bm.as_jax(rng.random(indices.shape))
#   r1 = jax.grad(sum_op(bm.event.csrmv))(
#     data, indices, indptr, events, shape=shape, transpose=transpose)
#   r2 = jax.grad(sum_op2(bm.event.csrmv_taichi))(
#     data, indices, indptr, events, shape=shape, transpose=transpose)
#   assert(bm.allclose(r1, r2))

#   # grad 'events'
#   r3 = jax.grad(sum_op(bm.event.csrmv), argnums=3)(
#     data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
#   r4 = jax.grad(sum_op2(bm.event.csrmv_taichi), argnums=3)(
#     data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
#   assert(bm.allclose(r3, r4))

#   r5 = jax.grad(sum_op(bm.event.csrmv), argnums=(0, 3))(
#     data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
#   r6 = jax.grad(sum_op2(bm.event.csrmv_taichi), argnums=(0, 3))(
#     data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
#   assert(bm.allclose(r5[0], r6[0]))
#   assert(bm.allclose(r5[1], r6[1]))

#   bm.clear_buffer_memory()

# def test_all():
#   for transpose in transposes:
#     for shape in shapes:
#       for homo_data in homo_datas:
#         test_homo(shape, transpose, homo_data)
#         test_homo_vmap(shape, transpose, homo_data)
#         test_homo_grad(shape, transpose, homo_data)

#   for transpose in transposes:
#     for shape in shapes:
#         test_heter(shape, transpose)
#         test_heter_vmap(shape, transpose)
#         test_heter_grad(shape, transpose)
# test_all()


### PYTEST
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

    r1 = bm.event.csrmv(homo_data, indices, indptr, events, shape=shape, transpose=transpose)
    r2 = bm.event.csrmv_taichi(homo_data, indices, indptr, events, shape=shape, transpose=transpose)

    assert (bm.allclose(r1, r2[0]))

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
    f1 = jax.vmap(partial(bm.event.csrmv, indices=indices, indptr=indptr, events=events,
                          shape=shape, transpose=transpose))
    f2 = jax.vmap(partial(bm.event.csrmv_taichi, indices=indices, indptr=indptr, events=events,
                          shape=shape, transpose=transpose))
    vmap_data = bm.as_jax([homo_data] * 10)
    self.assertTrue(bm.allclose(f1(vmap_data), f2(vmap_data)[0]))

    # vmap 'events'
    f3 = jax.vmap(partial(bm.event.csrmv, homo_data, indices, indptr,
                          shape=shape, transpose=transpose))
    f4 = jax.vmap(partial(bm.event.csrmv_taichi, homo_data, indices, indptr,
                          shape=shape, transpose=transpose))
    vmap_data = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.1
    self.assertTrue(bm.allclose(f3(vmap_data), f4(vmap_data)[0]))

    # vmap 'data' and 'events'
    f5 = jax.vmap(lambda dd, ee: bm.event.csrmv(dd, indices, indptr, ee, shape=shape, transpose=transpose))
    f6 = jax.vmap(lambda dd, ee: bm.event.csrmv_taichi(dd, indices, indptr, ee, shape=shape, transpose=transpose))

    vmap_data1 = bm.as_jax([homo_data] * 10)
    vmap_data2 = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.2
    self.assertTrue(bm.allclose(f5(vmap_data1, vmap_data2),
                                f6(vmap_data1, vmap_data2)[0]))

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
    r1 = jax.grad(sum_op(bm.event.csrmv))(
      homo_data, indices, indptr, events, shape=shape, transpose=transpose)
    r2 = jax.grad(sum_op2(bm.event.csrmv_taichi))(
      homo_data, indices, indptr, events, shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r1, r2))

    # grad 'events'
    r3 = jax.grad(sum_op(bm.event.csrmv), argnums=3)(
      homo_data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    r4 = jax.grad(sum_op2(bm.event.csrmv_taichi), argnums=3)(
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

    r1 = bm.event.csrmv(heter_data, indices, indptr, events,
                        shape=shape, transpose=transpose)
    r2 = bm.event.csrmv_taichi(heter_data, indices, indptr, events,
                               shape=shape, transpose=transpose)

    assert (bm.allclose(r1, r2[0]))

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
    f1 = jax.vmap(partial(bm.event.csrmv, indices=indices, indptr=indptr, events=events,
                          shape=shape, transpose=transpose))
    f2 = jax.vmap(partial(bm.event.csrmv_taichi, indices=indices, indptr=indptr, events=events,
                          shape=shape, transpose=transpose))
    vmap_data = bm.as_jax(rng.random((10, indices.shape[0])))
    self.assertTrue(bm.allclose(f1(vmap_data), f2(vmap_data)[0]))

    # vmap 'events'
    data = bm.as_jax(rng.random(indices.shape))
    f3 = jax.vmap(partial(bm.event.csrmv, data, indices, indptr,
                          shape=shape, transpose=transpose))
    f4 = jax.vmap(partial(bm.event.csrmv_taichi, data, indices, indptr,
                          shape=shape, transpose=transpose))
    vmap_data = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.1
    self.assertTrue(bm.allclose(f3(vmap_data), f4(vmap_data)[0]))

    # vmap 'data' and 'events'
    f5 = jax.vmap(lambda dd, ee: bm.event.csrmv(dd, indices, indptr, ee,
                                                shape=shape, transpose=transpose))
    f6 = jax.vmap(lambda dd, ee: bm.event.csrmv_taichi(dd, indices, indptr, ee,
                                                       shape=shape, transpose=transpose))
    vmap_data1 = bm.as_jax(rng.random((10, indices.shape[0])))
    vmap_data2 = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.2
    self.assertTrue(bm.allclose(f5(vmap_data1, vmap_data2),
                                f6(vmap_data1, vmap_data2)[0]))

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
    r1 = jax.grad(sum_op(bm.event.csrmv))(
      data, indices, indptr, events, shape=shape, transpose=transpose)
    r2 = jax.grad(sum_op2(bm.event.csrmv_taichi))(
      data, indices, indptr, events, shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r1, r2))

    # grad 'events'
    r3 = jax.grad(sum_op(bm.event.csrmv), argnums=3)(
      data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    r4 = jax.grad(sum_op2(bm.event.csrmv_taichi), argnums=3)(
      data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r3, r4))

    r5 = jax.grad(sum_op(bm.event.csrmv), argnums=(0, 3))(
      data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    r6 = jax.grad(sum_op2(bm.event.csrmv_taichi), argnums=(0, 3))(
      data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r5[0], r6[0]))
    self.assertTrue(bm.allclose(r5[1], r6[1]))

    bm.clear_buffer_memory()
