# -*- coding: utf-8 -*-

from functools import partial

import jax
import pytest
from absl.testing import parameterized
import platform
import brainpy as bp
import brainpy.math as bm

is_manual_test = False
if platform.system() == 'Windows' and not is_manual_test:
  pytest.skip('brainpy.math package may need manual tests.', allow_module_level=True)

cusparse_csr_matvec = partial(bm.sparse.csrmv, method='cusparse')
scalar_csr_matvec = partial(bm.sparse.csrmv, method='scalar')
vector_csr_matvec = partial(bm.sparse.csrmv, method='vector')


class Test_cusparse_csrmv(parameterized.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(Test_cusparse_csrmv, self).__init__(*args, **kwargs)

    print()
    bm.set_platform(platform)

  @parameterized.product(
    transpose=[True, False],
    shape=[(200, 200), (200, 100), (10, 1000), (2, 2000)],
    homo_data=[-1., 0., 1.]
  )
  def test_homo(self, transpose, shape, homo_data):
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)

    heter_data = bm.ones(indices.shape).value * homo_data

    vector = rng.random(shape[0] if transpose else shape[1])
    vector = bm.as_jax(vector)
    r1 = cusparse_csr_matvec(homo_data, indices, indptr, vector, shape=shape, transpose=transpose)
    r2 = cusparse_csr_matvec(heter_data, indices, indptr, vector, shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r1, r2))

    dense = bm.sparse.csr_to_dense(heter_data, indices, indptr, shape=shape)
    r3 = (vector @ dense) if transpose else (dense @ vector)
    self.assertTrue(bm.allclose(r1, r3))

    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(200, 200), (200, 100), (10, 1000), (2, 2000)],
    v=[-1., 0., 1.]
  )
  def test_homo_vmap(self, transpose, shape, v):
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    vector = rng.random(shape[0] if transpose else shape[1])
    vector = bm.as_jax(vector)

    heter_data = bm.ones((10, indices.shape[0])).value * v
    homo_data = bm.ones(10).value * v
    dense_data = jax.vmap(lambda a: bm.sparse.csr_to_dense(a, indices, indptr, shape=shape))(heter_data)

    f1 = partial(cusparse_csr_matvec, indices=indices, indptr=indptr, vector=vector,
                 shape=shape, transpose=transpose)
    f2 = lambda a: (a.T @ vector) if transpose else (a @ vector)

    r1 = jax.vmap(f1)(homo_data)
    r2 = jax.vmap(f1)(heter_data)
    self.assertTrue(bm.allclose(r1, r2))

    r3 = jax.vmap(f2)(dense_data)
    self.assertTrue(bm.allclose(r1, r3))

    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(200, 200), (200, 100), (10, 1000), (2, 2000)],
    homo_data=[-1., 0., 1.]
  )
  def test_homo_grad(self, transpose, shape, homo_data):
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    dense = bm.sparse.csr_to_dense(bm.ones(indices.shape).value,
                                   indices,
                                   indptr,
                                   shape=shape)
    vector = rng.random(shape[0] if transpose else shape[1])
    vector = bm.as_jax(vector)

    csr_f1 = jax.grad(lambda a: cusparse_csr_matvec(a, indices, indptr, vector,
                                                    shape=shape, transpose=transpose).sum(),
                      argnums=0)
    dense_f1 = jax.grad(lambda a: ((vector @ (dense * a)).sum()
                                   if transpose else
                                   ((dense * a) @ vector).sum()),
                        argnums=0)

    r1 = csr_f1(homo_data)
    r2 = dense_f1(homo_data)
    self.assertTrue(bm.allclose(r1, r2))

    csr_f2 = jax.grad(lambda v: cusparse_csr_matvec(homo_data, indices, indptr, v,
                                                    shape=shape, transpose=transpose).sum())
    dense_data = dense * homo_data
    dense_f2 = jax.grad(lambda v: ((v @ dense_data).sum() if transpose else (dense_data @ v).sum()))

    r3 = csr_f2(vector)
    r4 = dense_f2(vector)
    self.assertTrue(bm.allclose(r3, r4))

    csr_f3 = jax.grad(lambda a, v: cusparse_csr_matvec(a, indices, indptr, v,
                                                       shape=shape, transpose=transpose).sum(),
                      argnums=(0, 1))
    dense_f3 = jax.grad(lambda a, v: ((v @ (dense * a)).sum()
                                      if transpose else
                                      ((dense * a) @ v).sum()),
                        argnums=(0, 1))

    r5 = csr_f3(homo_data, vector)
    r6 = dense_f3(homo_data, vector)
    self.assertTrue(bm.allclose(r5[0], r6[0]))
    self.assertTrue(bm.allclose(r5[1], r6[1]))

    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(200, 200), (200, 100), (10, 1000), (2, 2000)],
  )
  def test_heter(self, transpose, shape):
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)

    heter_data = rng.random(indices.shape)
    heter_data = bm.as_jax(heter_data)

    vector = rng.random(shape[0] if transpose else shape[1])
    vector = bm.as_jax(vector)
    r1 = cusparse_csr_matvec(heter_data, indices, indptr, vector,
                             shape=shape, transpose=transpose)
    dense = bm.sparse.csr_to_dense(heter_data, indices, indptr, shape=shape)
    r2 = (vector @ dense) if transpose else (dense @ vector)
    self.assertTrue(bm.allclose(r1, r2))

    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(200, 200), (200, 100), (10, 1000), (2, 2000)]
  )
  def test_heter_vmap(self, transpose, shape):
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    vector = rng.random(shape[0] if transpose else shape[1])
    vector = bm.as_jax(vector)

    heter_data = rng.random((10, indices.shape[0]))
    heter_data = bm.as_jax(heter_data)
    dense_data = jax.vmap(lambda a: bm.sparse.csr_to_dense(a, indices, indptr,
                                                           shape=shape))(heter_data)

    f1 = partial(cusparse_csr_matvec, indices=indices, indptr=indptr, vector=vector,
                 shape=shape, transpose=transpose)
    f2 = lambda a: (a.T @ vector) if transpose else (a @ vector)

    r1 = jax.vmap(f1)(heter_data)
    r2 = jax.vmap(f2)(dense_data)
    self.assertTrue(bm.allclose(r1, r2))

    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(200, 200), (200, 100), (10, 1000), (2, 2000)]
  )
  def test_heter_grad(self, transpose, shape):
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    heter_data = rng.random(indices.shape)
    heter_data = bm.as_jax(heter_data)
    dense_data = bm.sparse.csr_to_dense(heter_data, indices, indptr, shape=shape)
    vector = rng.random(shape[0] if transpose else shape[1])
    vector = bm.as_jax(vector)

    csr_f1 = jax.grad(lambda a: cusparse_csr_matvec(a, indices, indptr, vector,
                                                    shape=shape,
                                                    transpose=transpose).sum(),
                      argnums=0)
    dense_f1 = jax.grad(lambda a: ((vector @ a).sum() if transpose else (a @ vector).sum()),
                        argnums=0)

    r1 = csr_f1(heter_data)
    r2 = dense_f1(dense_data)
    rows, cols = bm.sparse.csr_to_coo(indices, indptr)
    r2 = r2[rows, cols]
    self.assertTrue(bm.allclose(r1, r2))

    csr_f2 = jax.grad(lambda v: cusparse_csr_matvec(heter_data, indices, indptr, v,
                                                    shape=shape,
                                                    transpose=transpose).sum(),
                      argnums=0)
    dense_f2 = jax.grad(lambda v: ((v @ dense_data).sum() if transpose else (dense_data @ v).sum()),
                        argnums=0)
    r3 = csr_f2(vector)
    r4 = dense_f2(vector)
    self.assertTrue(bm.allclose(r3, r4))

    bm.clear_buffer_memory()


class Test_csrmv(parameterized.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(Test_csrmv, self).__init__(*args, **kwargs)

    print()
    bm.set_platform(platform)

  @parameterized.product(
    homo_data=[-1., 0., 0.1, 1.],
    shape=[(100, 200), (10, 1000), (2, 2000)],
  )
  def test_homo(self, shape, homo_data):
    conn = bp.conn.FixedProb(0.1)

    # matrix
    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    # vector
    rng = bm.random.RandomState(123)
    vector = rng.random(shape[1])
    vector = bm.as_jax(vector)

    # csrmv
    r1 = scalar_csr_matvec(homo_data, indices, indptr, vector, shape=shape)
    r2 = cusparse_csr_matvec(homo_data, indices, indptr, vector, shape=shape)
    r3 = vector_csr_matvec(homo_data, indices, indptr, vector, shape=shape)
    self.assertTrue(bm.allclose(r1, r2))
    self.assertTrue(bm.allclose(r1, r3))

    heter_data = bm.ones(indices.shape).to_jax() * homo_data
    r4 = scalar_csr_matvec(heter_data, indices, indptr, vector, shape=shape)
    r5 = cusparse_csr_matvec(heter_data, indices, indptr, vector, shape=shape)
    r6 = vector_csr_matvec(heter_data, indices, indptr, vector, shape=shape)
    self.assertTrue(bm.allclose(r1, r4))
    self.assertTrue(bm.allclose(r1, r5))
    self.assertTrue(bm.allclose(r1, r6))

    dense = bm.sparse.csr_to_dense(heter_data, indices, indptr, shape=shape)
    rdense = dense @ vector
    self.assertTrue(bm.allclose(r1, rdense))

    bm.clear_buffer_memory()

  @parameterized.product(
    shape=[(100, 200), (200, 100), (10, 1000), (2, 2000)]
  )
  def test_heter(self, shape):
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    heter_data = bm.as_jax(rng.random(indices.shape))
    vector = bm.as_jax(rng.random(shape[1]))

    r1 = scalar_csr_matvec(heter_data, indices, indptr, vector, shape=shape)
    r2 = cusparse_csr_matvec(heter_data, indices, indptr, vector, shape=shape)
    r3 = vector_csr_matvec(heter_data, indices, indptr, vector, shape=shape)

    dense = bm.sparse.csr_to_dense(heter_data, indices, indptr, shape=shape)
    r4 = dense @ vector
    self.assertTrue(bm.allclose(r1, r2))
    self.assertTrue(bm.allclose(r1, r3))
    self.assertTrue(bm.allclose(r1, r4))

    bm.clear_buffer_memory()

  @parameterized.product(
    shape=[(200, 200), (200, 100), (10, 1000), (2, 2000)]
  )
  def test_heter_grad(self, shape):
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    heter_data = rng.random(indices.shape)
    dense_data = bm.sparse.csr_to_dense(heter_data, indices, indptr, shape=shape)
    vector = rng.random(shape[1])

    csr_f1 = jax.grad(lambda a: cusparse_csr_matvec(a, indices, indptr, vector, shape=shape).sum())
    csr_f2 = jax.grad(lambda a: scalar_csr_matvec(a, indices, indptr, vector, shape=shape).sum())
    csr_f3 = jax.grad(lambda a: vector_csr_matvec(a, indices, indptr, vector, shape=shape).sum())
    dense_f1 = jax.grad(lambda a: (a @ vector).sum())

    r1 = csr_f1(heter_data)
    r2 = csr_f2(heter_data)
    r3 = csr_f3(heter_data)

    d1 = dense_f1(dense_data)
    rows, cols = bm.sparse.csr_to_coo(indices, indptr)
    d1 = d1[rows, cols]
    self.assertTrue(bm.allclose(r1, r2))
    self.assertTrue(bm.allclose(r1, r3))
    self.assertTrue(bm.allclose(r1, d1))

    # csr_f4 = jax.grad(lambda v: cusparse_csr_matvec(heter_data, indices, indptr, v, shape=shape).sum())
    # csr_f5 = jax.grad(lambda v: scalar_csr_matvec(heter_data, indices, indptr, v, shape=shape).sum())
    # csr_f6 = jax.grad(lambda v: vector_csr_matvec(heter_data, indices, indptr, v, shape=shape).sum())
    # dense_f2 = jax.grad(lambda v: (dense_data @ v).sum())
    # r4 = csr_f4(vector)
    # r5 = csr_f5(vector)
    # r6 = csr_f6(vector)
    # d2 = dense_f2(vector)
    # self.assertTrue(bm.allclose(r4, r5))
    # self.assertTrue(bm.allclose(r4, r6))
    # self.assertTrue(bm.allclose(r4, d2))

    bm.clear_buffer_memory()


