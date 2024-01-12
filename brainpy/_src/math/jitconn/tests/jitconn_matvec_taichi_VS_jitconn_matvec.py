# from jax_taichi import jax_taichi_call

import time
from functools import partial
import os

import brainpy as bp
import brainpy.math as bm
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import taichi as ti

bm.set_platform('gpu')

seed = 1234

shape = [
        1000, 
        2500, 
        5000, 
        10000, 
        25000, 
        37500, 
        50000
        ]
types = [
       'homo', 
       'uniform',
       'normal'
       ]
transpose = [
            True, 
            False
            ]
outdim_parallel = [
                  True,
                  False,
                  ]
bool_event = False
conn_prob = 0.05
homo_data = 1.
w_low = 0.
w_high = 1.
w_mu = 0.
w_sigma = 0.1

ITERATION = 100
if bm.get_platform() == 'cpu':
  ITERATION = 10

print(bm.get_platform())

@partial(jax.jit, static_argnums=(4, 5, 6))
def jitconn_matvec_homo_taichi(vector, homo_data, conn_prob, seed, shape, transpose, outdim_parallel):
  r = 0
  for i in range(ITERATION):
    r += bm.jitconn.mv_prob_homo_taichi(vector, homo_data, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)
  return r
  
@partial(jax.jit, static_argnums=(4, 5, 6))
def jitconn_matvec_homo(vector, homo_data, conn_prob, seed, shape, transpose, outdim_parallel):
  r = 0
  for i in range(ITERATION):
    r += bm.jitconn.mv_prob_homo(vector, homo_data, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)
  return r
  
@partial(jax.jit, static_argnums=(5, 6, 7))
def jitconn_matvec_uniform_taichi(vector, w_low, w_high, conn_prob, seed, shape, transpose, outdim_parallel):
  r = 0
  for i in range(ITERATION):
    r += bm.jitconn.mv_prob_uniform_taichi(vector, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)
  return r
  
@partial(jax.jit, static_argnums=(5, 6, 7))
def jitconn_matvec_uniform(vector, w_low, w_high, conn_prob, seed, shape, transpose, outdim_parallel):
  r = 0
  for i in range(ITERATION):
    r += bm.jitconn.mv_prob_uniform(vector, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)
  return r
  
@partial(jax.jit, static_argnums=(5, 6, 7))
def jitconn_matvec_normal_taichi(vector, w_mu, w_sigma, conn_prob, seed, shape, transpose, outdim_parallel):
  r = 0
  for i in range(ITERATION):
    r += bm.jitconn.mv_prob_normal_taichi(vector, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)
  return r
  
@partial(jax.jit, static_argnums=(5, 6, 7))
def jitconn_matvec_normal(vector, w_mu, w_sigma, conn_prob, seed, shape, transpose, outdim_parallel):
  r = 0
  for i in range(ITERATION):
    r += bm.jitconn.mv_prob_normal(vector, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)
  return r

def test_jitconn_matvec_homo(shape, transpose, outdim_parallel):
  rng = bm.random.RandomState(seed=seed)
  vector = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

  # groundtruth = bm.as_jax(vector, dtype=float) @ bm.as_jax(dense)

  result = jax.block_until_ready(jitconn_matvec_homo_taichi(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_homo_taichi(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_homo_taichi(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_homo_taichi(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_homo_taichi(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))

  time0 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo_taichi(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time1 = time.time()

  time2 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo_taichi(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time3 = time.time()

  time4 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo_taichi(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time5 = time.time()

  time6 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo_taichi(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time7 = time.time()

  time8 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo_taichi(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time9 = time.time()
  
  time10 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo_taichi(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time11 = time.time()
  
  time12 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo_taichi(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time13 = time.time()
  
  time14 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo_taichi(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time15 = time.time()
  
  time16 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo_taichi(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time17 = time.time()
  
  time18 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo_taichi(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time19 = time.time()
  

  result = jax.block_until_ready(jitconn_matvec_homo(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_homo(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_homo(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_homo(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_homo(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))

  time20 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time21 = time.time()

  time22 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time23 = time.time()

  time24 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time25 = time.time()

  time26 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time27 = time.time()

  time28 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time29 = time.time()
  
  time30 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time31 = time.time()
  
  time32 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time33 = time.time()
  
  time34 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time35 = time.time()
  
  time36 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time37 = time.time()
  
  time38 = time.time()
  result = jax.block_until_ready(jitconn_matvec_homo(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time39 = time.time()

  taichi_aot_time1 = (time1 - time0) * 1000
  taichi_aot_time2 = (time3 - time2) * 1000
  taichi_aot_time3 = (time5 - time4) * 1000
  taichi_aot_time4 = (time7 - time6) * 1000
  taichi_aot_time5 = (time9 - time8) * 1000
  taichi_aot_time6 = (time11 - time10) * 1000
  taichi_aot_time7 = (time13 - time12) * 1000
  taichi_aot_time8 = (time15 - time14) * 1000
  taichi_aot_time9 = (time17 - time16) * 1000
  taichi_aot_time10 = (time19 - time18) * 1000
  brainpy_time1 = (time21 - time20) * 1000
  brainpy_time2 = (time23 - time22) * 1000
  brainpy_time3 = (time25 - time24) * 1000
  brainpy_time4 = (time27 - time26) * 1000
  brainpy_time5 = (time29 - time28) * 1000
  brainpy_time6 = (time31 - time30) * 1000
  brainpy_time7 = (time33 - time32) * 1000
  brainpy_time8 = (time35 - time34) * 1000
  brainpy_time9 = (time37 - time36) * 1000
  brainpy_time10 = (time39 - time38) * 1000
  print('taichi_aot_1: ', taichi_aot_time1, 'ms')
  print('taichi_aot_3: ', taichi_aot_time3, 'ms')
  print('taichi_aot_5: ', taichi_aot_time5, 'ms')
  print('taichi_aot_7: ', taichi_aot_time7, 'ms')
  print('taichi_aot_9: ', taichi_aot_time9, 'ms')
  print('brainpylib_1: ', brainpy_time1, 'ms')
  print('brainpylib_3: ', brainpy_time3, 'ms')
  print('brainpylib_5: ', brainpy_time5, 'ms')
  print('brainpylib_7: ', brainpy_time7, 'ms')
  print('brainpylib_9: ', brainpy_time9, 'ms')


  return taichi_aot_time1, taichi_aot_time2, taichi_aot_time3, taichi_aot_time4, taichi_aot_time5,\
      taichi_aot_time6, taichi_aot_time7, taichi_aot_time8, taichi_aot_time9, taichi_aot_time10,\
      brainpy_time1, brainpy_time2, brainpy_time3, brainpy_time4, brainpy_time5, \
      brainpy_time6, brainpy_time7, brainpy_time8, brainpy_time9, brainpy_time10

def test_jitconn_matvec_uniform(shape, transpose, outdim_parallel):
  rng = bm.random.RandomState(seed=seed)
  events = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

  # groundtruth = bm.as_jax(vector, dtype=float) @ bm.as_jax(dense)

  result1 = jax.block_until_ready(jitconn_matvec_uniform_taichi(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_uniform_taichi(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_uniform_taichi(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_uniform_taichi(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_uniform_taichi(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))

  time0 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform_taichi(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time1 = time.time()

  time2 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform_taichi(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time3 = time.time()

  time4 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform_taichi(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time5 = time.time()

  time6 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform_taichi(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time7 = time.time()

  time8 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform_taichi(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time9 = time.time()
  
  time10 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform_taichi(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time11 = time.time()
  
  time12 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform_taichi(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time13 = time.time()
  
  time14 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform_taichi(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time15 = time.time()
  
  time16 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform_taichi(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time17 = time.time()
  
  time18 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform_taichi(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time19 = time.time()
  

  result = jax.block_until_ready(jitconn_matvec_uniform(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_uniform(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_uniform(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_uniform(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_uniform(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))

  time20 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time21 = time.time()

  time22 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time23 = time.time()

  time24 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time25 = time.time()

  time26 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time27 = time.time()

  time28 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time29 = time.time()
  
  time30 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time31 = time.time()
  
  time32 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time33 = time.time()
  
  time34 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time35 = time.time()
  
  time36 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time37 = time.time()
  
  time38 = time.time()
  result = jax.block_until_ready(jitconn_matvec_uniform(events, w_low, w_high, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time39 = time.time()

  taichi_aot_time1 = (time1 - time0) * 1000
  taichi_aot_time2 = (time3 - time2) * 1000
  taichi_aot_time3 = (time5 - time4) * 1000
  taichi_aot_time4 = (time7 - time6) * 1000
  taichi_aot_time5 = (time9 - time8) * 1000
  taichi_aot_time6 = (time11 - time10) * 1000
  taichi_aot_time7 = (time13 - time12) * 1000
  taichi_aot_time8 = (time15 - time14) * 1000
  taichi_aot_time9 = (time17 - time16) * 1000
  taichi_aot_time10 = (time19 - time18) * 1000
  brainpy_time1 = (time21 - time20) * 1000
  brainpy_time2 = (time23 - time22) * 1000
  brainpy_time3 = (time25 - time24) * 1000
  brainpy_time4 = (time27 - time26) * 1000
  brainpy_time5 = (time29 - time28) * 1000
  brainpy_time6 = (time31 - time30) * 1000
  brainpy_time7 = (time33 - time32) * 1000
  brainpy_time8 = (time35 - time34) * 1000
  brainpy_time9 = (time37 - time36) * 1000
  brainpy_time10 = (time39 - time38) * 1000
  print('taichi_aot_1: ', taichi_aot_time1, 'ms')
  print('taichi_aot_3: ', taichi_aot_time3, 'ms')
  print('taichi_aot_5: ', taichi_aot_time5, 'ms')
  print('taichi_aot_7: ', taichi_aot_time7, 'ms')
  print('taichi_aot_9: ', taichi_aot_time9, 'ms')
  print('brainpylib_1: ', brainpy_time1, 'ms')
  print('brainpylib_3: ', brainpy_time3, 'ms')
  print('brainpylib_5: ', brainpy_time5, 'ms')
  print('brainpylib_7: ', brainpy_time7, 'ms')
  print('brainpylib_9: ', brainpy_time9, 'ms')


  return taichi_aot_time1, taichi_aot_time2, taichi_aot_time3, taichi_aot_time4, taichi_aot_time5,\
      taichi_aot_time6, taichi_aot_time7, taichi_aot_time8, taichi_aot_time9, taichi_aot_time10,\
      brainpy_time1, brainpy_time2, brainpy_time3, brainpy_time4, brainpy_time5, \
      brainpy_time6, brainpy_time7, brainpy_time8, brainpy_time9, brainpy_time10

def test_jitconn_matvec_normal(shape, transpose, outdim_parallel):
  rng = bm.random.RandomState(seed=seed)
  events = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

  # groundtruth = bm.as_jax(vector, dtype=float) @ bm.as_jax(dense)

  result = jax.block_until_ready(jitconn_matvec_normal_taichi(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_normal_taichi(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_normal_taichi(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_normal_taichi(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_normal_taichi(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))

  time0 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal_taichi(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time1 = time.time()

  time2 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal_taichi(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time3 = time.time()

  time4 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal_taichi(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time5 = time.time()

  time6 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal_taichi(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time7 = time.time()

  time8 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal_taichi(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time9 = time.time()
  
  time10 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal_taichi(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time11 = time.time()
  
  time12 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal_taichi(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time13 = time.time()
  
  time14 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal_taichi(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time15 = time.time()
  
  time16 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal_taichi(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time17 = time.time()
  
  time18 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal_taichi(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time19 = time.time()
  

  result = jax.block_until_ready(jitconn_matvec_normal(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_normal(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_normal(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_normal(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  result = jax.block_until_ready(jitconn_matvec_normal(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))

  time20 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time21 = time.time()

  time22 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time23 = time.time()

  time24 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time25 = time.time()

  time26 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time27 = time.time()

  time28 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time29 = time.time()
  
  time30 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time31 = time.time()
  
  time32 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time33 = time.time()
  
  time34 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time35 = time.time()
  
  time36 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time37 = time.time()
  
  time38 = time.time()
  result = jax.block_until_ready(jitconn_matvec_normal(events, w_mu, w_sigma, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time39 = time.time()

  taichi_aot_time1 = (time1 - time0) * 1000
  taichi_aot_time2 = (time3 - time2) * 1000
  taichi_aot_time3 = (time5 - time4) * 1000
  taichi_aot_time4 = (time7 - time6) * 1000
  taichi_aot_time5 = (time9 - time8) * 1000
  taichi_aot_time6 = (time11 - time10) * 1000
  taichi_aot_time7 = (time13 - time12) * 1000
  taichi_aot_time8 = (time15 - time14) * 1000
  taichi_aot_time9 = (time17 - time16) * 1000
  taichi_aot_time10 = (time19 - time18) * 1000
  brainpy_time1 = (time21 - time20) * 1000
  brainpy_time2 = (time23 - time22) * 1000
  brainpy_time3 = (time25 - time24) * 1000
  brainpy_time4 = (time27 - time26) * 1000
  brainpy_time5 = (time29 - time28) * 1000
  brainpy_time6 = (time31 - time30) * 1000
  brainpy_time7 = (time33 - time32) * 1000
  brainpy_time8 = (time35 - time34) * 1000
  brainpy_time9 = (time37 - time36) * 1000
  brainpy_time10 = (time39 - time38) * 1000
  print('taichi_aot_1: ', taichi_aot_time1, 'ms')
  print('taichi_aot_3: ', taichi_aot_time3, 'ms')
  print('taichi_aot_5: ', taichi_aot_time5, 'ms')
  print('taichi_aot_7: ', taichi_aot_time7, 'ms')
  print('taichi_aot_9: ', taichi_aot_time9, 'ms')
  print('brainpylib_1: ', brainpy_time1, 'ms')
  print('brainpylib_3: ', brainpy_time3, 'ms')
  print('brainpylib_5: ', brainpy_time5, 'ms')
  print('brainpylib_7: ', brainpy_time7, 'ms')
  print('brainpylib_9: ', brainpy_time9, 'ms')


  return taichi_aot_time1, taichi_aot_time2, taichi_aot_time3, taichi_aot_time4, taichi_aot_time5,\
      taichi_aot_time6, taichi_aot_time7, taichi_aot_time8, taichi_aot_time9, taichi_aot_time10,\
      brainpy_time1, brainpy_time2, brainpy_time3, brainpy_time4, brainpy_time5, \
      brainpy_time6, brainpy_time7, brainpy_time8, brainpy_time9, brainpy_time10

def test_jitconn_matvec(shape, _type, transpose, outdim_parallel):
  print('shape: ', shape, ' type: ', _type, ' transpose: ', transpose, ' outdim_parallel: ', outdim_parallel)
  if _type == 'homo':
    return test_jitconn_matvec_homo(shape, transpose, outdim_parallel)
  elif _type == 'uniform':
    return test_jitconn_matvec_uniform(shape, transpose, outdim_parallel)
  elif _type == 'normal':
    return test_jitconn_matvec_normal(shape, transpose, outdim_parallel)
  else:
    raise ValueError

PATH = os.path.dirname(os.path.abspath(__file__))

# init dataframe
df = pd.DataFrame(columns=['shape[0]', 'shape[1]', 'backend', 'type', 'transpose', 'outdim_parallel', 'bool_event',
                            'taichi aot time1(ms)', 'taichi aot time2(ms)', 'taichi aot time3(ms)', 'taichi aot time4(ms)', 'taichi aot time5(ms)',
                           'taichi aot time6(ms)', 'taichi aot time7(ms)', 'taichi aot time8(ms)', 'taichi aot time9(ms)', 'taichi aot time10(ms)',
                           'brainpy time1(ms)', 'brainpy time2(ms)', 'brainpy time3(ms)', 'brainpy time4(ms)', 'brainpy time5(ms)',
                           'brainpy time6(ms)', 'brainpy time7(ms)', 'brainpy time8(ms)', 'brainpy time9(ms)', 'brainpy time10(ms)'])

### RECTANGULAR MATRIX
if (bm.get_platform() == 'cpu'):
  for shape1 in shape:
    for shape2 in shape:
      for _type in types:
        for _outdim_parallel in outdim_parallel:
          for _transpose in transpose:
            taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5,\
                taichi_aot_time_6, taichi_aot_time_7, taichi_aot_time_8, taichi_aot_time_9, taichi_aot_time_10,\
                  brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, \
                  brainpy_time_6, brainpy_time_7, brainpy_time_8, brainpy_time_9, brainpy_time_10 = test_jitconn_matvec((shape1, shape2), _type, _transpose, _outdim_parallel)
            # append to dataframe
            df.loc[df.shape[0]] = [shape1, shape2, 'cpu', _type, _transpose, _outdim_parallel, bool_event,
                                  taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5,
                                    taichi_aot_time_6, taichi_aot_time_7, taichi_aot_time_8, taichi_aot_time_9, taichi_aot_time_10,
                                    brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, 
                                    brainpy_time_6, brainpy_time_7, brainpy_time_8, brainpy_time_9, brainpy_time_10]
  df.to_csv(f'{PATH}/jitconn_matvec_cpu.csv', index=False)

if (bm.get_platform() == 'gpu'):
  for shape1 in shape:
    for shape2 in shape:
      for _type in types:
        for _outdim_parallel in outdim_parallel:
          for _transpose in transpose:
            taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5,\
                taichi_aot_time_6, taichi_aot_time_7, taichi_aot_time_8, taichi_aot_time_9, taichi_aot_time_10,\
                  brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, \
                  brainpy_time_6, brainpy_time_7, brainpy_time_8, brainpy_time_9, brainpy_time_10 = test_jitconn_matvec((shape1, shape2), _type, _transpose, _outdim_parallel)
            # append to dataframe
            df.loc[df.shape[0]] = [shape1, shape2, 'cpu', _type, _transpose, _outdim_parallel, bool_event,
                                  taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5,
                                    taichi_aot_time_6, taichi_aot_time_7, taichi_aot_time_8, taichi_aot_time_9, taichi_aot_time_10,
                                    brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, 
                                    brainpy_time_6, brainpy_time_7, brainpy_time_8, brainpy_time_9, brainpy_time_10]
  df.to_csv(f'{PATH}/jitconn_matvec_gpu.csv', index=False)
