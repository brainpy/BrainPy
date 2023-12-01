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

vector_csr_matvec = partial(bm.sparse.csrmv, method='vector')

# transposes=[True, False]
# homo_datas=[-1., 0., 0.1, 1.]
# shapes=[(100, 200), (10, 1000), (2, 2000)]

# def test_homo(transpose, shape, homo_data):
#     print(f'test_homo: transpose = {transpose} shape = {shape}, homo_data = {homo_data}')
#     conn = bp.conn.FixedProb(0.1)

#     # matrix
#     indices, indptr = conn(*shape).require('pre2post')
#     indices = bm.as_jax(indices)
#     indptr = bm.as_jax(indptr)
#     # vector
#     rng = bm.random.RandomState(123)
#     vector = rng.random(shape[0] if transpose else shape[1])
#     vector = bm.as_jax(vector)

#     r1 = vector_csr_matvec(homo_data, indices, indptr, vector, shape=shape, transpose=transpose)
#     r2 = bm.sparse.csrmv_taichi(homo_data, indices, indptr, vector, shape=shape, transpose=transpose)
#     assert(bm.allclose(r1, r2[0]))

#     bm.clear_buffer_memory()

# def test_homo_vmap(transpose, shape, v):
#     print(f'test_homo_vmap: transpose = {transpose} shape = {shape}, v = {v}')
#     rng = bm.random.RandomState()
#     conn = bp.conn.FixedProb(0.1)

#     indices, indptr = conn(*shape).require('pre2post')
#     indices = bm.as_jax(indices)
#     indptr = bm.as_jax(indptr)
#     vector = rng.random(shape[0] if transpose else shape[1])
#     vector = bm.as_jax(vector)

#     heter_data = bm.ones((10, indices.shape[0])).value * v
#     homo_data = bm.ones(10).value * v
#     dense_data = jax.vmap(lambda a: bm.sparse.csr_to_dense(a, indices, indptr, shape=shape))(heter_data)

#     f1 = partial(vector_csr_matvec, indices=indices, indptr=indptr, vector=vector,
#                  shape=shape, transpose=transpose)
#     f2 = partial(bm.sparse.csrmv_taichi, indices=indices, indptr=indptr, vector=vector,
#                  shape=shape, transpose=transpose)
#     r1 = jax.vmap(f1)(homo_data)
#     r2 = jax.vmap(f1)(homo_data)
#     assert(bm.allclose(r1, r2[0]))

#     bm.clear_buffer_memory()

# def test_homo_grad(transpose, shape, v):
#     print(f'test_homo_grad: transpose = {transpose} shape = {shape}, v = {v}')
#     rng = bm.random.RandomState()
#     conn = bp.conn.FixedProb(0.1)

#     indices, indptr = conn(*shape).require('pre2post')
#     indices = bm.as_jax(indices)
#     indptr = bm.as_jax(indptr)
#     dense = bm.sparse.csr_to_dense(bm.ones(indices.shape).value,
#                                    indices,
#                                    indptr,
#                                    shape=shape)
#     vector = rng.random(shape[0] if transpose else shape[1])
#     vector = bm.as_jax(vector)


#     # print('grad data start')
#     # grad 'data'
#     r1 = jax.grad(sum_op(vector_csr_matvec))(
#        homo_data, indices, indptr, vector, shape=shape, transpose=transpose)
#     r2 = jax.grad(sum_op2(bm.sparse.csrmv_taichi))(
#        homo_data, indices, indptr, vector, shape=shape, transpose = transpose)
    
#     # csr_f1 = jax.grad(lambda a: vector_csr_matvec(a, indices, indptr, vector,
#     #                                                 shape=shape, transpose=transpose).sum(),
#     #                   argnums=0)
#     # csr_f2 = jax.grad(lambda a: bm.sparse.csrmv_taichi(a, indices, indptr, vector,
#     #                                                 shape=shape, transpose=transpose)[0].sum(),
#     #                   argnums=0)
#     # r1 = csr_f1(homo_data)
#     # r2 = csr_f2(homo_data)
#     assert(bm.allclose(r1, r2))

#     # print('grad vector start')
#     # grad 'vector'
#     r3 = jax.grad(sum_op(vector_csr_matvec), argnums=3)(
#        homo_data, indices, indptr, vector.astype(float), shape=shape, transpose=transpose)
#     r4 = jax.grad(sum_op2(bm.sparse.csrmv_taichi), argnums=3)(
#        homo_data, indices, indptr, vector.astype(float), shape=shape, transpose=transpose)
#     # csr_f3 = jax.grad(lambda v: vector_csr_matvec(homo_data, indices, indptr, v,
#     #                                                 shape=shape, transpose=transpose).sum())
#     # csr_f4 = jax.grad(lambda v: bm.sparse.csrmv_taichi(homo_data, indices, indptr, v,
#     #                                                 shape=shape, transpose=transpose)[0].sum())
#     # r3 = csr_f3(vector)
#     # r4 = csr_f4(vector)
#     assert(bm.allclose(r3, r4))

#     # csr_f5 = jax.grad(lambda a, v: vector_csr_matvec(a, indices, indptr, v,
#     #                                                    shape=shape, transpose=transpose).sum(),
#     #                   argnums=(0, 1))
#     # csr_f6 = jax.grad(lambda a, v: bm.sparse.csrmv_taichi(a, indices, indptr, v,
#     #                                                    shape=shape, transpose=transpose)[0].sum(),
#     #                   argnums=(0, 1))
#     # r5 = csr_f5(homo_data, vector)
#     # r6 = csr_f6(homo_data, vector)
#     # assert(bm.allclose(r5[0], r6[0]))
#     # assert(bm.allclose(r5[1], r6[1]))

#     bm.clear_buffer_memory()

# def test_heter(shape):
#     print(f'test_homo: shape = {shape}')
#     rng = bm.random.RandomState()
#     conn = bp.conn.FixedProb(0.1)

#     indices, indptr = conn(*shape).require('pre2post')
#     indices = bm.as_jax(indices)
#     indptr = bm.as_jax(indptr)
#     heter_data = bm.as_jax(rng.random(indices.shape))
#     vector = bm.as_jax(rng.random(shape[1]))

#     r1 = vector_csr_matvec(heter_data, indices, indptr, vector, shape=shape)
#     r2 = bm.sparse.csrmv_taichi(heter_data, indices, indptr, vector, shape=shape)

#     assert(bm.allclose(r1, r2[0]))

#     bm.clear_buffer_memory()

# for transpose in transposes:
#     for shape in shapes:
#         for homo_data in homo_datas:
#             test_homo(transpose, shape, homo_data)

# for shape in shapes:
#     test_heter(shape)

# for transpose in transposes:
#     for shape in shapes:
#         for homo_data in homo_datas:
#             test_homo_vmap(transpose, shape, homo_data)

# for transpose in transposes:
#     for shape in shapes:
#         for homo_data in homo_datas:
#             test_homo_grad(transpose, shape, homo_data)

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
    print(f'test_homo: transpose = {transpose} shape = {shape}, homo_data = {homo_data}')
    conn = bp.conn.FixedProb(0.1)

    # matrix
    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    # vector
    rng = bm.random.RandomState(123)
    vector = rng.random(shape[0] if transpose else shape[1])
    vector = bm.as_jax(vector)

    r1 = vector_csr_matvec(homo_data, indices, indptr, vector, shape=shape, transpose=transpose)
    r2 = bm.sparse.csrmv_taichi(homo_data, indices, indptr, vector, shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r1, r2[0]))

    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(200, 200), (200, 100), (10, 1000), (2, 2000)],
    v=[-1., 0., 1.]
  )
  def test_homo_vmap(self, transpose, shape, v):
    print(f'test_homo_vmap: transpose = {transpose} shape = {shape}, v = {v}')
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

    f1 = partial(vector_csr_matvec, indices=indices, indptr=indptr, vector=vector,
                 shape=shape, transpose=transpose)
    f2 = partial(bm.sparse.csrmv_taichi, indices=indices, indptr=indptr, vector=vector,
                 shape=shape, transpose=transpose)
    r1 = jax.vmap(f1)(homo_data)
    r2 = jax.vmap(f1)(homo_data)
    self.assertTrue(bm.allclose(r1, r2[0]))

    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(200, 200), (200, 100), (10, 1000), (2, 2000)],
    homo_data=[-1., 0., 1.]
  )
  def test_homo_grad(self, transpose, shape, homo_data):
    print(f'test_homo_grad: transpose = {transpose} shape = {shape}, homo_data = {homo_data}')
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


    # print('grad data start')
    # grad 'data'
    r1 = jax.grad(sum_op(vector_csr_matvec))(
       homo_data, indices, indptr, vector, shape=shape, transpose=transpose)
    r2 = jax.grad(sum_op2(bm.sparse.csrmv_taichi))(
       homo_data, indices, indptr, vector, shape=shape, transpose = transpose)
    
    # csr_f1 = jax.grad(lambda a: vector_csr_matvec(a, indices, indptr, vector,
    #                                                 shape=shape, transpose=transpose).sum(),
    #                   argnums=0)
    # csr_f2 = jax.grad(lambda a: bm.sparse.csrmv_taichi(a, indices, indptr, vector,
    #                                                 shape=shape, transpose=transpose)[0].sum(),
    #                   argnums=0)
    # r1 = csr_f1(homo_data)
    # r2 = csr_f2(homo_data)
    self.assertTrue(bm.allclose(r1, r2))

    # print('grad vector start')
    # grad 'vector'
    r3 = jax.grad(sum_op(vector_csr_matvec), argnums=3)(
       homo_data, indices, indptr, vector.astype(float), shape=shape, transpose=transpose)
    r4 = jax.grad(sum_op2(bm.sparse.csrmv_taichi), argnums=3)(
       homo_data, indices, indptr, vector.astype(float), shape=shape, transpose=transpose)
    # csr_f3 = jax.grad(lambda v: vector_csr_matvec(homo_data, indices, indptr, v,
    #                                                 shape=shape, transpose=transpose).sum())
    # csr_f4 = jax.grad(lambda v: bm.sparse.csrmv_taichi(homo_data, indices, indptr, v,
    #                                                 shape=shape, transpose=transpose)[0].sum())
    # r3 = csr_f3(vector)
    # r4 = csr_f4(vector)
    self.assertTrue(bm.allclose(r3, r4))

    # csr_f5 = jax.grad(lambda a, v: vector_csr_matvec(a, indices, indptr, v,
    #                                                    shape=shape, transpose=transpose).sum(),
    #                   argnums=(0, 1))
    # csr_f6 = jax.grad(lambda a, v: bm.sparse.csrmv_taichi(a, indices, indptr, v,
    #                                                    shape=shape, transpose=transpose)[0].sum(),
    #                   argnums=(0, 1))
    # r5 = csr_f5(homo_data, vector)
    # r6 = csr_f6(homo_data, vector)
    # assert(bm.allclose(r5[0], r6[0]))
    # assert(bm.allclose(r5[1], r6[1]))
    bm.clear_buffer_memory()

  @parameterized.product(
    transpose=[True, False],
    shape=[(200, 200), (200, 100), (10, 1000), (2, 2000)],
  )
  def test_heter(self, transpose, shape):
    print(f'test_homo: transpose = {transpose} shape = {shape}')
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    
    heter_data = bm.as_jax(rng.random(indices.shape))
    heter_data = bm.as_jax(heter_data)

    vector = rng.random(shape[0] if transpose else shape[1])
    vector = bm.as_jax(vector)

    r1 = vector_csr_matvec(heter_data, indices, indptr, vector, shape=shape)
    r2 = bm.sparse.csrmv_taichi(heter_data, indices, indptr, vector, shape=shape)

    self.assertTrue(bm.allclose(r1, r2[0]))

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

    f1 = partial(vector_csr_matvec, indices=indices, indptr=indptr, vector=vector,
                 shape=shape, transpose=transpose)
    f2 = partial(bm.sparse.csrmv_taichi, indices=indices, indptr=indptr, vector=vector,
                 shape=shape, transpose=transpose)
    r1 = jax.vmap(f1)(heter_data)
    r2 = jax.vmap(f1)(heter_data)
    self.assertTrue(bm.allclose(r1, r2[0]))

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

    # grad 'data'
    r1 = jax.grad(sum_op(vector_csr_matvec))(
       heter_data, indices, indptr, vector, shape=shape, transpose=transpose)
    r2 = jax.grad(sum_op2(bm.sparse.csrmv_taichi))(
       heter_data, indices, indptr, vector, shape=shape, transpose = transpose)
    self.assertTrue(bm.allclose(r1, r2))

    # grad 'vector'
    r3 = jax.grad(sum_op(vector_csr_matvec), argnums=3)(
       heter_data, indices, indptr, vector.astype(float), shape=shape, transpose=transpose)
    r4 = jax.grad(sum_op2(bm.sparse.csrmv_taichi), argnums=3)(
       heter_data, indices, indptr, vector.astype(float), shape=shape, transpose=transpose)
    self.assertTrue(bm.allclose(r3, r4))

    bm.clear_buffer_memory()
