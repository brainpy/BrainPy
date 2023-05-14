# -*- coding: utf-8 -*-

import unittest
import brainpylib
import jax.numpy as jnp

import brainpy as bp
import brainpy.math as bm


class Test_csr_matvec(unittest.TestCase):
  def __init__(self, *args, platform='cpu', **kwargs):
    super(Test_csr_matvec, self).__init__(*args, **kwargs)

    print()
    bm.set_platform(platform)

  def _test_homo(self, shape, homo_data):
    print(f'{self._test_homo.__name__}: shape = {shape}, homo_data = {homo_data}')

    conn = bp.conn.FixedProb(0.1)

    # matrix
    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    # vector
    rng = bm.random.RandomState(123)
    vector = rng.random(shape[1])
    vector = bm.as_jax(vector)
    r1 = brainpylib.csr_matvec(homo_data, indices, indptr, vector, shape=shape)

    heter_data = bm.ones(indices.shape).to_jax() * homo_data
    r2 = brainpylib.csr_matvec(heter_data, indices, indptr, vector, shape=shape)
    self.assertTrue(jnp.allclose(r1, r2))

    r3 = brainpylib.cusparse_csr_matvec(heter_data, indices, indptr, vector, shape=shape)
    self.assertTrue(jnp.allclose(r1, r3))

    dense = brainpylib.csr_to_dense(heter_data, indices, indptr, shape=shape)
    r4 = dense @ vector
    self.assertTrue(jnp.allclose(r1, r4))

    bm.clear_buffer_memory()

  def test_homo(self):
    for v in [-1., 0., 0.1, 1.]:
      for shape in [(100, 200),
                    (10, 1000),
                    (2, 2000)]:
        self._test_homo(shape, v)

  def _test_heter(self, shape):
    print(f'{self._test_heter.__name__}: shape = {shape}')
    rng = bm.random.RandomState()
    conn = bp.conn.FixedProb(0.1)

    indices, indptr = conn(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    heter_data = rng.random(indices.shape)
    heter_data = bm.as_jax(heter_data)
    vector = rng.random(shape[1])
    vector = bm.as_jax(vector)

    r1 = brainpylib.csr_matvec(heter_data, indices, indptr, vector, shape=shape)

    dense = brainpylib.csr_to_dense(heter_data, indices, indptr, shape=shape)
    r2 = dense @ vector
    self.assertTrue(jnp.allclose(r1, r2))

    r3 = brainpylib.cusparse_csr_matvec(heter_data, indices, indptr, vector, shape=shape)
    self.assertTrue(jnp.allclose(r1, r3))

    bm.clear_buffer_memory()

  def test_csr_matvec_heter_1(self):
    for shape in [(100, 200),
                  (200, 100),
                  (10, 1000),
                  (2, 2000)
                  ]:
      self._test_heter(shape)

