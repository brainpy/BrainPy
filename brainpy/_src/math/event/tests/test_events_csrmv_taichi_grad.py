# -*- coding: utf-8 -*-


from functools import partial

import jax

import brainpy as bp
import brainpy.math as bm
import platform

import pytest

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

transposes = [True, False]
shapes = [(100, 200),
                  (200, 200),
                  (200, 100),
                  (10, 1000),
                  (2, 10000),
                  (1000, 10),
                  (100000, 2)]
homo_datas = [-1., 0., 1.]

def test_homo_grad(shape, transpose, homo_data):
    print(f'test_homo_grad: shape = {shape}, transpose = {transpose}, homo_data = {homo_data}')

    rng = bm.random.RandomState()
    indices, indptr = bp.conn.FixedProb(0.4)(*shape).require('pre2post')
    indices = bm.as_jax(indices)
    indptr = bm.as_jax(indptr)
    events = bm.as_jax(rng.random(shape[0] if transpose else shape[1])) < 0.1
    dense_conn = bm.sparse.csr_to_dense(bm.ones(indices.shape).value, indices, indptr, shape=shape)

    # grad 'data'
    r1 = jax.grad(sum_op(bm.event.csrmv))(homo_data, 
                                          indices, 
                                          indptr, 
                                          events, 
                                          shape=shape, 
                                          transpose=transpose)
    
    r2 = jax.grad(sum_op2(bm.event.csrmv_taichi))(homo_data, 
                                                indices, 
                                                indptr, 
                                                events, 
                                                shape=shape, 
                                                transpose=transpose)

    assert(bm.allclose(r1, r2))

for transpose in transposes:
   for shape in shapes:
      for homo_data in homo_datas:
         test_homo_grad(shape, transpose, homo_data)