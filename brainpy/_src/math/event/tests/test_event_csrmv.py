# -*- coding: utf-8 -*-


from functools import partial

import jax
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm
import platform

import pytest

is_manual_test = False
if platform.system() == 'Windows' and not is_manual_test:
  pytest.skip('brainpy.math package may need manual tests.', allow_module_level=True)


def sum_op(op):
  def func(*args, **kwargs):
    r = op(*args, **kwargs)
    return r.sum()

  return func


class Test_event_csr_matvec(parameterized.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(Test_event_csr_matvec, self).__init__(*args, **kwargs)
    bm.set_platform(platform)
    print()

  @parameterized.named_parameters(
    dict(
      testcase_name=f'transpose={transpose}, shape={shape}, homo_data={homo_data}',
      transpose=transpose,
      shape=shape,
      homo_data=homo_data,
    )
    for transpose in [True, False]
    for shape in [(100, 200),
                  (200, 200),
                  (200, 100),
                  (10, 1000),
                  (2, 10000),
                  (1000, 10),
                  (10000, 2)]
    for homo_data in [-1., 0., 1.]
  )
  def test_homo(self, shape, transpose, homo_data):
    print(f'test_homo: shape = {shape}, transpose = {transpose}, homo_data = {homo_data}')

    rng = bm.random.RandomState()
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    events = rng.random(shape[0] if transpose else shape[1]) < 0.1
    heter_data = bm.ones(indices.shape) * homo_data

    r1 = bm.event.csrmv(homo_data, indices, indptr, events, shape=shape, transpose=transpose)
    r2 = bm.event.csrmv(heter_data, indices, indptr, events, shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r1, r2))

    r3 = bm.event.csrmv(homo_data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r1, r3))

    dense = bm.sparse.csr_to_dense(heter_data, indices, indptr, shape=shape)
    r4 = (events @ dense) if transpose else (dense @ events)
    self.assertTrue(bm.allclose(r1, r4))

    r5 = bm.event.csrmv(heter_data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r1, r5))

    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    dict(
      testcase_name=f'transpose={transpose}, shape={shape}, homo_data={homo_data}',
      transpose=transpose,
      shape=shape,
      homo_data=homo_data,
    )
    for transpose in [True, False]
    for shape in [(100, 200),
                  (200, 200),
                  (200, 100),
                  (10, 1000),
                  (2, 10000),
                  (1000, 10),
                  (100000, 2)]
    for homo_data in [-1., 0., 1.]
  )
  def test_homo_vmap(self, shape, transpose, homo_data):
    print(f'test_homo_vamp: shape = {shape}, transpose = {transpose}, homo_data = {homo_data}')

    rng = bm.random.RandomState()
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')

    # vmap 'data'
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
    f1 = jax.vmap(partial(bm.event.csrmv, indices=indices, indptr=indptr, events=events,
                          shape=shape, transpose=transpose))
    f2 = jax.vmap(
      partial(partial(bm.sparse.csrmv, method='cusparse'), indices=indices, indptr=indptr, vector=events.astype(float),
              shape=shape, transpose=transpose))
    vmap_data = bm.as_jax([homo_data] * 10)
    self.assertTrue(bm.allclose(f1(vmap_data), f2(vmap_data)))

    # vmap 'events'
    f3 = jax.vmap(partial(bm.event.csrmv, homo_data, indices, indptr,
                          shape=shape, transpose=transpose))
    f4 = jax.vmap(partial(partial(bm.sparse.csrmv, method='cusparse'), homo_data, indices, indptr,
                          shape=shape, transpose=transpose))
    vmap_data = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.1
    self.assertTrue(bm.allclose(f3(vmap_data), f4(vmap_data.astype(float))))

    # vmap 'data' and 'events'
    f5 = jax.vmap(lambda dd, ee: bm.event.csrmv(dd, indices, indptr, ee, shape=shape, transpose=transpose))
    f6 = jax.vmap(lambda dd, ee: bm.sparse.csrmv(dd, indices, indptr, ee, shape=shape, transpose=transpose,
                                                 method='cusparse'))
    vmap_data1 = bm.as_jax([homo_data] * 10)
    vmap_data2 = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.2
    self.assertTrue(bm.allclose(f5(vmap_data1, vmap_data2),
                                 f6(vmap_data1, vmap_data2.astype(float))))

    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    dict(
      testcase_name=f'transpose={transpose},shape={shape},homo_data={homo_data}',
      homo_data=homo_data,
      shape=shape,
      transpose=transpose,
    )
    for transpose in [True, False]
    for shape in [(100, 200),
                  (200, 200),
                  (200, 100),
                  (10, 1000),
                  (2, 10000),
                  (1000, 10),
                  (100000, 2)]
    for homo_data in [-1., 0., 1.]
  )
  def test_homo_grad(self, shape, transpose, homo_data):
    print(f'test_homo_grad: shape = {shape}, transpose = {transpose}, homo_data = {homo_data}')

    rng = bm.random.RandomState()
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
    dense_conn = bm.sparse.csr_to_dense(bm.ones(indices.shape).value, indices, indptr, shape=shape)

    # grad 'data'
    r1 = jax.grad(sum_op(bm.event.csrmv))(
      homo_data, indices, indptr, events, shape=shape, transpose=transpose)
    r2 = jax.grad(sum_op(partial(bm.sparse.csrmv, method='cusparse')))(
      homo_data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r1, r2))
    r3 = jax.grad(sum_op(lambda a: (events @ (dense_conn * a) if transpose else
                                    ((dense_conn * a) @ events))))(homo_data)
    self.assertTrue(bm.allclose(r1, r3))

    # grad 'events'
    r4 = jax.grad(sum_op(bm.event.csrmv), argnums=3)(
      homo_data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    r5 = jax.grad(sum_op(partial(bm.sparse.csrmv, method='cusparse')), argnums=3)(
      homo_data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    r6 = jax.grad(sum_op(lambda e: (e @ (dense_conn * homo_data) if transpose else
                                    ((dense_conn * homo_data) @ e))))(events.astype(float))
    self.assertTrue(bm.allclose(r4, r5))
    self.assertTrue(bm.allclose(r4, r6))

    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    dict(
      testcase_name=f'transpose={transpose}, shape={shape}',
      shape=shape,
      transpose=transpose,
    )
    for transpose in [True, False]
    for shape in [(100, 200),
                  (200, 200),
                  (200, 100),
                  (10, 1000),
                  (2, 10000),
                  (1000, 10),
                  (10000, 2)]
  )
  def test_heter(self, shape, transpose):
    print(f'test_heter: shape = {shape}, transpose = {transpose}')

    rng = bm.random.RandomState()
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
    heter_data = bm.as_jax(rng.random(indices.shape))

    r1 = bm.event.csrmv(heter_data, indices, indptr, events,
                        shape=shape, transpose=transpose)
    r2 = partial(bm.sparse.csrmv, method='cusparse')(heter_data, indices, indptr, events.astype(float),
                                                     shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r1, r2))

    dense = bm.sparse.csr_to_dense(heter_data, indices, indptr, shape=shape)
    r3 = (events @ dense) if transpose else (dense @ events)
    self.assertTrue(bm.allclose(r1, r3))

    r4 = bm.event.csrmv(heter_data, indices, indptr, events.astype(float),
                        shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r1, r4))

    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    dict(
      testcase_name=f"transpose={transpose}, shape={shape}",
      shape=shape,
      transpose=transpose,
    )
    for transpose in [True, False]
    for shape in [(100, 200),
                  (200, 200),
                  (200, 100),
                  (10, 1000),
                  (2, 10000),
                  (1000, 10),
                  (100000, 2)]
  )
  def test_heter_vmap(self, shape, transpose):
    print(f'test_heter_vamp: shape = {shape}, transpose = {transpose}')

    rng = bm.random.RandomState()
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)

    # vmap 'data'
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
    f1 = jax.vmap(partial(bm.event.csrmv, indices=indices, indptr=indptr, events=events,
                          shape=shape, transpose=transpose))
    f2 = jax.vmap(
      partial(partial(bm.sparse.csrmv, method='cusparse'), indices=indices, indptr=indptr, vector=events.astype(float),
              shape=shape, transpose=transpose))
    vmap_data = bm.as_jax(rng.random((10, indices.shape[0])))
    self.assertTrue(bm.allclose(f1(vmap_data), f2(vmap_data)))

    # vmap 'events'
    data = bm.as_jax(rng.random(indices.shape))
    f3 = jax.vmap(partial(bm.event.csrmv, data, indices, indptr,
                          shape=shape, transpose=transpose))
    f4 = jax.vmap(partial(partial(bm.sparse.csrmv, method='cusparse'), data, indices, indptr,
                          shape=shape, transpose=transpose))
    vmap_data = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.1
    self.assertTrue(bm.allclose(f3(vmap_data), f4(vmap_data.astype(float))))

    # vmap 'data' and 'events'
    f5 = jax.vmap(lambda dd, ee: bm.event.csrmv(dd, indices, indptr, ee,
                                                shape=shape, transpose=transpose))
    f6 = jax.vmap(lambda dd, ee: partial(bm.sparse.csrmv, method='cusparse')(dd, indices, indptr, ee,
                                                                             shape=shape, transpose=transpose))
    vmap_data1 = bm.as_jax(rng.random((10, indices.shape[0])))
    vmap_data2 = bm.as_jax(rng.random((10, shape[0] if transpose else shape[1]))) < 0.2
    self.assertTrue(bm.allclose(f5(vmap_data1, vmap_data2),
                                 f6(vmap_data1, vmap_data2.astype(float))))

    bm.clear_buffer_memory()

  @parameterized.named_parameters(
    dict(testcase_name=f'transpose={transpose},shape={shape}',
         shape=shape,
         transpose=transpose,
         )
    for transpose in [True, False]
    for shape in [(100, 200),
                  (200, 200),
                  (200, 100),
                  (10, 1000),
                  (2, 10000),
                  (1000, 10),
                  (100000, 2)]
  )
  def test_heter_grad(self, shape, transpose):
    print(f'test_heter_grad: shape = {shape}, transpose = {transpose}')

    rng = bm.random.RandomState()
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
    r2 = jax.grad(sum_op(partial(bm.sparse.csrmv, method='cusparse')))(
      data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r1, r2))

    dense_data = bm.sparse.csr_to_dense(data, indices, indptr, shape=shape)
    r3 = jax.grad(sum_op(lambda a: ((events @ a) if transpose else
                                    (a @ events))))(dense_data)
    rows, cols = bm.sparse.csr_to_coo(indices, indptr)
    r3 = r3[rows, cols]
    self.assertTrue(bm.allclose(r1, r3))

    # grad 'events'
    r4 = jax.grad(sum_op(bm.event.csrmv), argnums=3)(
      data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    r5 = jax.grad(sum_op(partial(bm.sparse.csrmv, method='cusparse')), argnums=3)(
      data, indices, indptr, events.astype(float), shape=shape, transpose=transpose)
    r6 = jax.grad(sum_op(lambda e: ((e @ dense_data) if transpose else
                                    (dense_data @ e))))(events.astype(float))
    self.assertTrue(bm.allclose(r4, r5))
    self.assertTrue(bm.allclose(r4, r6))

    bm.clear_buffer_memory()
