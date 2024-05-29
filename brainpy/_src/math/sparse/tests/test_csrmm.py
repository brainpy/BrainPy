# -*- coding: utf-8 -*-

from functools import partial

import jax
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm

# bm.set_platform('gpu')

seed = 1234


def sum_op(op):
  def func(*args, **kwargs):
    r = op(*args, **kwargs)
    return r.sum()

  return func


class Test_csrmm(parameterized.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(Test_csrmm, self).__init__(*args, **kwargs)

    print()
    bm.set_platform(platform)

  @parameterized.product(
    transpose=[True, False],
    shape=[(50, 50, 50), (100, 50, 100), (10, 1000, 10), (2, 2000, 2)],
    homo_data=[-1., 1.]
  )
  def test_homo(self, transpose, shape, homo_data):
    print(f'test_homo: transpose: {transpose} shape = {shape}')
    conn = bp.conn.FixedProb(0.3)

    # csr matrix
    indices, indptr = conn(shape[1], shape[0]).require('pre2post') if transpose else conn(shape[0],
                                                                                          shape[1]).require(
      'pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    # matrix
    rng = bm.random.RandomState(seed=seed)
    matrix = rng.random((shape[1], shape[2])) < 0.1
    matrix = bm.as_jax(matrix)

    heter_data = bm.ones(indices.shape) * homo_data

    dense = bm.sparse.csr_to_dense(heter_data, indices, indptr,
                                   shape=(shape[1], shape[0]) if transpose else (shape[0], shape[1]))

    r1 = (dense.T @ matrix) if transpose else (dense @ matrix)
    r2 = bm.sparse.csrmm(homo_data, indices, indptr, matrix,
                         shape=(shape[1], shape[0]) if transpose else (shape[0], shape[1]), transpose=transpose)
    c = bm.allclose(r1, r2, equal_nan=True)
    if not c:
      print(r1 - r2)
    self.assertTrue(c)

    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(50, 50, 50), (100, 50, 100), (10, 1000, 10), (2, 2000, 2)],
    homo_data=[-1., 1.]
  )
  def test_homo_vmap(self, transpose, shape, homo_data):
    print(f'test_homo_vmap: transpose: {transpose} shape = {shape}')
    conn = bp.conn.FixedProb(0.3)

    # csr matrix
    indices, indptr = conn(shape[1], shape[0]).require('pre2post') if transpose else conn(shape[0],
                                                                                          shape[1]).require(
      'pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    # matrix
    rng = bm.random.RandomState(seed=seed)
    matrix = rng.random((shape[1], shape[2])) < 0.1
    matrix = bm.as_jax(matrix)

    heter_data = bm.ones((10, indices.shape[0])) * homo_data
    dense = jax.vmap(lambda a: bm.sparse.csr_to_dense(a, indices, indptr,
                                                      shape=(shape[1], shape[0]) if transpose else (
                                                        shape[0], shape[1])))(heter_data)

    # vmap 'data'
    f1 = jax.vmap(lambda a: (a.T @ matrix) if transpose else (a @ matrix))
    f2 = jax.vmap(partial(bm.sparse.csrmm, indices=indices, indptr=indptr, matrix=matrix,
                          shape=(shape[1], shape[0]) if transpose else (shape[0], shape[1]), transpose=transpose))
    vmap_data = bm.as_jax([homo_data] * 10)

    r1 = f1(dense)
    r2 = f2(vmap_data)
    self.assertTrue(bm.allclose(r1, r2))

    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(50, 50, 50), (100, 50, 100), (10, 1000, 10), (2, 2000, 2)],
    homo_data=[-1., 1.]
  )
  def test_homo_grad(self, transpose, shape, homo_data):
    print(f'test_homo_grad: transpose: {transpose} shape = {shape}')
    rng = bm.random.RandomState(seed=seed)
    conn = bp.conn.FixedProb(0.3)

    # csr matrix
    indices, indptr = conn(shape[1], shape[0]).require('pre2post') if transpose else conn(shape[0],
                                                                                          shape[1]).require(
      'pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    dense = bm.sparse.csr_to_dense(bm.ones(indices.shape).value,
                                   indices,
                                   indptr,
                                   shape=(shape[1], shape[0]) if transpose else (shape[0], shape[1]))

    heter_data = bm.as_jax(rng.random((indices.shape)))
    # matrix
    matrix = rng.random((shape[1], shape[2])) < 0.1
    matrix = bm.as_jax(matrix)

    # grad data
    dense_f1 = jax.grad(lambda a: (((dense.T * a) @ matrix).sum()
                                   if transpose else
                                   ((dense * a) @ matrix).sum()),
                        argnums=0)
    r1 = dense_f1(homo_data)
    r2 = jax.grad(sum_op(bm.sparse.csrmm))(
      bm.asarray([homo_data]), indices, indptr, matrix, shape=(shape[1], shape[0]) if transpose else (shape[0], shape[1]),
      transpose=transpose)

    self.assertTrue(bm.allclose(r1, r2))

    # grad events matrix
    dense_f2 = jax.grad(lambda m: (((dense.T * homo_data) @ m).sum()
                                   if transpose else
                                   ((dense * homo_data) @ m).sum()),
                        argnums=0)
    r3 = dense_f2(matrix.astype(float))
    r4 = jax.grad(sum_op(bm.sparse.csrmm), argnums=3)(
      bm.asarray([homo_data]), indices, indptr, matrix.astype(float),
      shape=(shape[1], shape[0]) if transpose else (shape[0], shape[1]), transpose=transpose)

    self.assertTrue(bm.allclose(r3, r4))

    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(50, 50, 50), (100, 50, 100), (10, 1000, 10), (2, 2000, 2)],
  )
  def test_heter(self, transpose, shape):
    print(f'test_homo: transpose: {transpose} shape = {shape}')
    conn = bp.conn.FixedProb(0.3)

    # csr matrix
    indices, indptr = conn(shape[1], shape[0]).require('pre2post') if transpose else conn(shape[0],
                                                                                          shape[1]).require(
      'pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    # matrix
    rng = bm.random.RandomState(seed=seed)
    matrix = rng.random((shape[1], shape[2]))
    matrix = bm.as_jax(matrix)

    heter_data = bm.as_jax(rng.random(indices.shape))

    dense = bm.sparse.csr_to_dense(heter_data, indices, indptr,
                                   shape=(shape[1], shape[0]) if transpose else (shape[0], shape[1]))

    r1 = (dense.T @ matrix) if transpose else (dense @ matrix)
    r2 = bm.sparse.csrmm(heter_data, indices, indptr, matrix,
                         shape=(shape[1], shape[0]) if transpose else (shape[0], shape[1]), transpose=transpose)
    print(r2)
    print(r1.shape, '-', r2.shape)
    c = bm.allclose(r1, r2, equal_nan=True)
    if not c:
      print(r1 - r2)
    self.assertTrue(c)

    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(50, 50, 50), (100, 50, 100), (10, 1000, 10), (2, 2000, 2)],
  )
  def test_heter_vmap(self, transpose, shape):
    print(f'test_homo_vmap: transpose: {transpose} shape = {shape}')
    conn = bp.conn.FixedProb(0.3)

    # csr matrix
    indices, indptr = conn(shape[1], shape[0]).require('pre2post') if transpose else conn(shape[0],
                                                                                          shape[1]).require(
      'pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    # matrix
    rng = bm.random.RandomState(seed=seed)
    matrix = rng.random((shape[1], shape[2]))
    matrix = bm.as_jax(matrix)

    heter_data = bm.as_jax(rng.random((10, indices.shape[0])))
    dense = jax.vmap(lambda a: bm.sparse.csr_to_dense(a, indices, indptr,
                                                      shape=(shape[1], shape[0]) if transpose else (
                                                        shape[0], shape[1])))(heter_data)

    f1 = lambda a: (a.T @ matrix) if transpose else (a @ matrix)
    f2 = partial(bm.sparse.csrmm, indices=indices, indptr=indptr, matrix=matrix,
                 shape=(shape[1], shape[0]) if transpose else (shape[0], shape[1]), transpose=transpose)
    r1 = jax.vmap(f1)(dense)
    r2 = jax.vmap(f2)(heter_data)

    self.assertTrue(bm.allclose(r1, r2, equal_nan=True))

  @parameterized.product(
    transpose=[True, False],
    shape=[(50, 50, 50), (100, 50, 100), (10, 1000, 10), (2, 2000, 2)],
  )
  def test_heter_grad(self, transpose, shape):
    print(f'test_homo_grad: transpose: {transpose} shape = {shape}')
    rng = bm.random.RandomState(seed=seed)
    conn = bp.conn.FixedProb(0.3)

    # csr matrix
    indices, indptr = conn(shape[1], shape[0]).require('pre2post') if transpose else conn(shape[0],
                                                                                          shape[1]).require(
      'pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)

    heter_data = bm.as_jax(rng.random((indices.shape)))
    dense = bm.sparse.csr_to_dense(heter_data,
                                   indices,
                                   indptr,
                                   shape=(shape[1], shape[0]) if transpose else (shape[0], shape[1]))
    # matrix
    matrix = rng.random((shape[1], shape[2]))
    matrix = bm.as_jax(matrix)

    # grad data
    dense_f1 = jax.grad(lambda a: ((a.T @ matrix).sum()
                                   if transpose else
                                   (a @ matrix).sum()),
                        argnums=0)
    r1 = dense_f1(dense)
    r2 = jax.grad(sum_op(bm.sparse.csrmm))(
      heter_data, indices, indptr, matrix, shape=(shape[1], shape[0]) if transpose else (shape[0], shape[1]),
      transpose=transpose
    )
    rows, cols = bm.sparse.csr_to_coo(indices, indptr)
    if transpose:
      r1 = r1[cols, rows]
    else:
      r1 = r1[rows, cols]
    print(r1 - r2)

    self.assertTrue(bm.allclose(r1, r2))

    # grad matrix
    dense_f2 = jax.grad(lambda m: ((dense.T @ m).sum()
                                   if transpose else
                                   (dense @ m).sum()))
    r3 = dense_f2(matrix)
    r4 = jax.grad(sum_op(bm.sparse.csrmm), argnums=3)(
      heter_data, indices, indptr, matrix.astype(float),
      shape=(shape[1], shape[0]) if transpose else (shape[0], shape[1]), transpose=transpose
    )

    self.assertTrue(bm.allclose(r3, r4))

    bm.clear_buffer_memory()
