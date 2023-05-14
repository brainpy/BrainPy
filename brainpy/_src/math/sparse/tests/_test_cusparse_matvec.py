# -*- coding: utf-8 -*-

import unittest

import brainpy as bp
import brainpy.math as bm

import jax
import jax.numpy as jnp
from functools import partial
import brainpylib


class Test_cusparse_csr_matvec(unittest.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(Test_cusparse_csr_matvec, self).__init__(*args, **kwargs)

    print()
    bm.set_platform(platform)

  def _test_homo(self, transpose, shape, homo_data):
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)

    heter_data = bm.ones(indices.shape).value * homo_data

    vector = rng.random(shape[0] if transpose else shape[1])
    vector = bm.as_jax(vector)
    r1 = brainpylib.cusparse_csr_matvec(homo_data, indices, indptr, vector, shape=shape, transpose=transpose)
    r2 = brainpylib.cusparse_csr_matvec(heter_data, indices, indptr, vector, shape=shape, transpose=transpose)
    self.assertTrue(jnp.allclose(r1, r2))

    dense = brainpylib.csr_to_dense(heter_data, indices, indptr, shape=shape)
    r3 = (vector @ dense) if transpose else (dense @ vector)
    self.assertTrue(jnp.allclose(r1, r3))

    bm.clear_buffer_memory()

  def _test_homo_vmap(self, transpose, shape, v):
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    vector = rng.random(shape[0] if transpose else shape[1])
    vector = bm.as_jax(vector)

    heter_data = bm.ones((10, indices.shape[0])).value * v
    homo_data = bm.ones(10).value * v
    dense_data = jax.vmap(lambda a: brainpylib.csr_to_dense(a, indices, indptr, shape=shape))(heter_data)

    f1 = partial(brainpylib.cusparse_csr_matvec, indices=indices, indptr=indptr, vector=vector,
                 shape=shape, transpose=transpose)
    f2 = lambda a: (a.T @ vector) if transpose else (a @ vector)

    r1 = jax.vmap(f1)(homo_data)
    r2 = jax.vmap(f1)(heter_data)
    self.assertTrue(jnp.allclose(r1, r2))

    r3 = jax.vmap(f2)(dense_data)
    self.assertTrue(jnp.allclose(r1, r3))

    bm.clear_buffer_memory()

  def _test_homo_grad(self, transpose, shape, homo_data):
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    dense = brainpylib.csr_to_dense(bm.ones(indices.shape).value,
                                    indices,
                                    indptr,
                                    shape=shape)
    vector = rng.random(shape[0] if transpose else shape[1])
    vector = bm.as_jax(vector)

    csr_f1 = jax.grad(lambda a: brainpylib.cusparse_csr_matvec(a, indices, indptr, vector,
                                                               shape=shape, transpose=transpose).sum(),
                      argnums=0)
    dense_f1 = jax.grad(lambda a: ((vector @ (dense * a)).sum()
                                   if transpose else
                                   ((dense * a) @ vector).sum()),
                        argnums=0)

    r1 = csr_f1(homo_data)
    r2 = dense_f1(homo_data)
    self.assertTrue(jnp.allclose(r1, r2))

    csr_f2 = jax.grad(lambda v: brainpylib.cusparse_csr_matvec(homo_data, indices, indptr, v,
                                                               shape=shape, transpose=transpose).sum(),
                      argnums=0)
    dense_data = dense * homo_data
    dense_f2 = jax.grad(lambda v: ((v @ dense_data).sum() if transpose else (dense_data @ v).sum()),
                        argnums=0)

    r3 = csr_f2(vector)
    r4 = dense_f2(vector)
    self.assertTrue(jnp.allclose(r3, r4))

    csr_f3 = jax.grad(lambda a, v: brainpylib.cusparse_csr_matvec(a, indices, indptr, v,
                                                                  shape=shape, transpose=transpose).sum(),
                      argnums=(0, 1))
    dense_f3 = jax.grad(lambda a, v: ((v @ (dense * a)).sum()
                                      if transpose else
                                      ((dense * a) @ v).sum()),
                        argnums=(0, 1))

    r5 = csr_f3(homo_data, vector)
    r6 = dense_f3(homo_data, vector)
    self.assertTrue(jnp.allclose(r5[0], r6[0]))
    self.assertTrue(jnp.allclose(r5[1], r6[1]))

    bm.clear_buffer_memory()

  def test_homo(self):
    for transpose in [True, False]:
      for shape in [(200, 200),
                    (200, 100),
                    (10, 1000),
                    (2, 2000)]:
        for homo_data in [-1.,
                          0.,
                          1.]:
          print(f'shape = {shape}, homo data = {homo_data}, transpose = {transpose}')

          self._test_homo(transpose, shape, homo_data)

  def test_homo_vmap(self):
    for transpose in [True, False]:
      for shape in [(200, 200),
                    (200, 100),
                    (10, 1000),
                    (2, 2000)]:
        for v in [-1.,
                  0.,
                  1.]:
          print(f'shape = {shape}, homo data = {v}')
          self._test_homo_vmap(transpose, shape, v)

  def test_homo_grad(self):
    for transpose in [True, False]:
      for shape in [(200, 200),
                    (200, 100),
                    (10, 1000),
                    (2, 2000)]:
        for v in [-1., 0., 1.]:
          print(f'shape = {shape}, homo data = {v}')
          self._test_homo_grad(transpose, shape, v)

  def _test_heter(self, transpose, shape):
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)

    heter_data = rng.random(indices.shape)
    heter_data = bm.as_jax(heter_data)

    vector = rng.random(shape[0] if transpose else shape[1])
    vector = bm.as_jax(vector)
    r1 = brainpylib.cusparse_csr_matvec(heter_data, indices, indptr, vector,
                                        shape=shape, transpose=transpose)
    dense = brainpylib.csr_to_dense(heter_data, indices, indptr, shape=shape)
    r2 = (vector @ dense) if transpose else (dense @ vector)
    self.assertTrue(jnp.allclose(r1, r2))

    bm.clear_buffer_memory()

  def _test_heter_vmap(self, transpose, shape):
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    vector = rng.random(shape[0] if transpose else shape[1])
    vector = bm.as_jax(vector)

    heter_data = rng.random((10, indices.shape[0]))
    heter_data = bm.as_jax(heter_data)
    dense_data = jax.vmap(lambda a: brainpylib.csr_to_dense(a, indices, indptr,
                                                            shape=shape))(heter_data)

    f1 = partial(brainpylib.cusparse_csr_matvec, indices=indices, indptr=indptr, vector=vector,
                 shape=shape, transpose=transpose)
    f2 = lambda a: (a.T @ vector) if transpose else (a @ vector)

    r1 = jax.vmap(f1)(heter_data)
    r2 = jax.vmap(f2)(dense_data)
    self.assertTrue(jnp.allclose(r1, r2))

    bm.clear_buffer_memory()

  def _test_heter_grad(self, transpose, shape):
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    heter_data = rng.random(indices.shape)
    heter_data = bm.as_jax(heter_data)
    dense_data = brainpylib.csr_to_dense(heter_data, indices, indptr, shape=shape)
    vector = rng.random(shape[0] if transpose else shape[1])
    vector = bm.as_jax(vector)

    csr_f1 = jax.grad(lambda a: brainpylib.cusparse_csr_matvec(a, indices, indptr, vector,
                                                               shape=shape,
                                                               transpose=transpose).sum(),
                      argnums=0)
    dense_f1 = jax.grad(lambda a: ((vector @ a).sum() if transpose else (a @ vector).sum()),
                        argnums=0)

    r1 = csr_f1(heter_data)
    r2 = dense_f1(dense_data)
    rows, cols = brainpylib.csr_to_coo(indices, indptr)
    r2 = r2[rows, cols]
    self.assertTrue(jnp.allclose(r1, r2))

    csr_f2 = jax.grad(lambda v: brainpylib.cusparse_csr_matvec(heter_data, indices, indptr, v,
                                                               shape=shape,
                                                               transpose=transpose).sum(),
                      argnums=0)
    dense_f2 = jax.grad(lambda v: ((v @ dense_data).sum() if transpose else (dense_data @ v).sum()),
                        argnums=0)
    r3 = csr_f2(vector)
    r4 = dense_f2(vector)
    self.assertTrue(jnp.allclose(r3, r4))

    bm.clear_buffer_memory()

  def test_heter(self):
    for transpose in [True, False]:
      for shape in [(200, 200),
                    (200, 100),
                    (10, 1000),
                    (2, 2000)]:
        print(f'shape = {shape}, transpose = {transpose}')
        self._test_heter(transpose, shape)

  def test_heter_vmap(self):
    for transpose in [True, False]:
      for shape in [(200, 200),
                    (200, 100),
                    (10, 1000),
                    (2, 2000)
                    ]:
        print(f'shape = {shape}, transpose = {transpose}')
        self._test_heter_vmap(transpose, shape)

  def test_heter_grad(self):
    for transpose in [True, False]:
      for shape in [(200, 200),
                    (200, 100),
                    (10, 1000),
                    (2, 2000)
                    ]:
        print(f'shape = {shape}, transpose = {transpose}')
        self._test_heter_grad(transpose, shape)
