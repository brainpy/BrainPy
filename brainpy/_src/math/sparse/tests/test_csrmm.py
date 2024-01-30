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
        homo_data=[-1., 0., 1.]
    )
    def test_homo(self, transpose, shape, homo_data):
        print(f'test_homo: transpose: {transpose} shape = {shape}, homo_data = {homo_data}')
        conn = bp.conn.FixedProb(0.3)

        # csr matrix
        indices, indptr = conn(shape[1], shape[0]).require('pre2post') if transpose else conn(shape[0],
                                                                                              shape[1]).require(
            'pre2post')
        indices = bm.as_jax(indices)
        indptr = bm.as_jax(indptr)
        # matrix
        rng = bm.random.RandomState(seed=seed)
        matrix = rng.random((shape[2], shape[1]) if transpose else (shape[1], shape[2]))
        matrix = bm.as_jax(matrix)

        heter_data = bm.ones(indices.shape).value * homo_data
        dense = bm.sparse.csr_to_dense(heter_data, indices, indptr,
                                       shape=(shape[1], shape[0]) if transpose else (shape[0], shape[1]))

        r1 = (matrix @ dense) if transpose else (dense @ matrix)
        r2 = bm.sparse.csrmm(homo_data, indices, indptr, matrix, shape=shape, transpose=transpose)
        c = bm.allclose(r1, r2)
        if not c:
            print(r1 - r2)
        self.assertTrue(c)

        bm.clear_buffer_memory()

    @parameterized.product(
        transpose=[True, False],
        shape=[(50, 50, 50), (100, 50, 100), (10, 1000, 10), (2, 2000, 2)],
        homo_data=[-1., 0., 1.]
    )
    def test_homo_vmap(self, transpose, shape, homo_data):
        print(f'test_homo_vmap: transpose: {transpose} shape = {shape}, homo_data = {homo_data}')
        conn = bp.conn.FixedProb(0.3)

        # csr matrix
        indices, indptr = conn(shape[1], shape[0]).require('pre2post') if transpose else conn(shape[0],
                                                                                              shape[1]).require(
            'pre2post')
        indices = bm.as_jax(indices)
        indptr = bm.as_jax(indptr)
        # matrix
        rng = bm.random.RandomState(seed=seed)
        matrix = rng.random((shape[2], shape[1]) if transpose else (shape[1], shape[2]))
        matrix = bm.as_jax(matrix)

        heter_data = bm.ones((10, indices.shape[0])).value * homo_data
        homo_data = bm.ones(10).value * homo_data
        dense = jax.vmap(lambda a: bm.sparse.csr_to_dense(a, indices, indptr, shape=(shape[1], shape[0]) if transpose else (shape[0], shape[1])))(heter_data)

        f1 = lambda a: (matrix @ a) if transpose else (a @ matrix)
        f2 = partial(bm.sparse.csrmm, indices=indices, indptr=indptr, matrix=matrix,
                     shape=shape, transpose=transpose)
        r1 = jax.vmap(f1)(dense)
        r2 = jax.vmap(f2)(homo_data)

        self.assertTrue(bm.allclose(r1, r2))


    @parameterized.product(
        transpose=[True, False],
        shape=[(50, 50, 50), (100, 50, 100), (10, 1000, 10), (2, 2000, 2)],
        homo_data=[-1., 0., 1.]
    )
    def test_homo_grad(self, transpose, shape, homo_data):
        print(f'test_homo_grad: transpose: {transpose} shape = {shape}, homo_data = {homo_data}')
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
        # matrix
        rng = bm.random.RandomState(seed=seed)
        matrix = rng.random((shape[2], shape[1]) if transpose else (shape[1], shape[2]))
        matrix = bm.as_jax(matrix)

        # grad data
        dense_f1 = jax.grad(lambda a: ((matrix @ (dense * a)).sum()
                                       if transpose else
                                       ((dense * a) @ matrix).sum()),
                            argnums=0)
        r1 = dense_f1(homo_data)
        r2 = jax.grad(sum_op(bm.sparse.csrmm))(
            homo_data, indices, indptr, matrix, shape=shape, transpose=transpose
        )

        self.assertTrue(bm.allclose(r1, r2))

        # grad matrix
        dense_data = dense * homo_data
        dense_f2 = jax.grad(lambda m: ((m @ dense_data).sum()
                                       if transpose else
                                       (dense_data @ m).sum()))
        r3 = dense_f2(matrix)
        r4 = jax.grad(sum_op(bm.sparse.csrmm), argnums=3)(
            homo_data, indices, indptr, matrix.astype(float), shape=shape, transpose=transpose
        )

        self.assertTrue(bm.allclose(r3, r4))

        # grad both
        dense_f3 = jax.grad(lambda a, m: ((m @ (dense * a)).sum()
                                          if transpose else
                                          ((dense * a) @ m).sum()),
                            argnums=(0, 1))
        r5 = dense_f3(homo_data, matrix)
        r6 = jax.grad(sum_op(bm.sparse.csrmm), argnums=(0, 3))(
            homo_data, indices, indptr, matrix.astype(float), shape=shape, transpose=transpose
        )

        self.assertTrue(bm.allclose(r5[0], r6[0]))
        self.assertTrue(bm.allclose(r5[1], r6[1]))

        bm.clear_buffer_memory()
