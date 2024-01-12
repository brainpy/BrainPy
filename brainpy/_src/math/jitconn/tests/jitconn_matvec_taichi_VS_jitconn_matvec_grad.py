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

bm.set_platform('cpu')

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
bool_event = False
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

def sum_op(op):
  def func(*args, **kwargs):
    r = op(*args, **kwargs)[0]
    return r.sum()

  return func

@partial(jax.jit, static_argnums=(4, 5, 6))
def jitconn_matvec_homo_taichi_grad(vector, homo_data, conn_prob, seed, shape, transpose, outdim_parallel):
  r = 0
  for i in range(ITERATION):
    r += jax.grad(sum_op(bm.jitconn.mv_prob_homo_taichi), argnums=0)(
      vector, homo_data, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )
  return r
  
@partial(jax.jit, static_argnums=(4, 5, 6))
def jitconn_matvec_homo_grad(vector, homo_data, conn_prob, seed, shape, transpose, outdim_parallel):
  r = 0
  for i in range(ITERATION):
    r += jax.grad(sum_op(bm.jitconn.mv_prob_homo), argnums=0)(
      vector, homo_data, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )
  return r
  
@partial(jax.jit, static_argnums=(5, 6, 7))
def jitconn_matvec_uniform_taichi_grad(vector, w_low, w_high, conn_prob, seed, shape, transpose, outdim_parallel):
  r = 0
  for i in range(ITERATION):
    r += jax.grad(sum_op(bm.jitconn.mv_prob_uniform_taichi), argnums=0)(
      vector, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )
  return r
  
@partial(jax.jit, static_argnums=(5, 6, 7))
def jitconn_matvec_uniform_grad(vector, w_low, w_high, conn_prob, seed, shape, transpose, outdim_parallel):
  r = 0
  for i in range(ITERATION):
    r += jax.grad(sum_op(bm.jitconn.mv_prob_uniform), argnums=0)(
      vector, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )
  return r
  
@partial(jax.jit, static_argnums=(5, 6, 7))
def jitconn_matvec_normal_taichi_grad(vector, w_mu, w_sigma, conn_prob, seed, shape, transpose, outdim_parallel):
  r = 0
  for i in range(ITERATION):
    r += jax.grad(sum_op(bm.jitconn.mv_prob_normal_taichi), argnums=0)(
      vector, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )
  return r
  
@partial(jax.jit, static_argnums=(5, 6, 7))
def jitconn_matvec_normal_grad(vector, w_mu, w_sigma, conn_prob, seed, shape, transpose, outdim_parallel):
  r = 0
  for i in range(ITERATION):
    r += jax.grad(sum_op(bm.jitconn.mv_prob_normal), argnums=0)(
      vector, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel
    )
  return r

def test_jitconn_matvec_homo_cpu(shape, transpose, outdim_parallel):
  rng = bm.random.RandomState(seed=seed)
  vector = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

  # groundtruth = bm.as_jax(vector, dtype=float) @ bm.as_jax(dense)

  result1 = jax.block_until_ready(jitconn_matvec_homo_taichi_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  # time.sleep(2)

  time0 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_homo_taichi_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time1 = time.time()
  # time.sleep(2)

  time2 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_homo_taichi_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time3 = time.time()
  # time.sleep(2)

  time4 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_homo_taichi_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time5 = time.time()
  # time.sleep(2)

  time6 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_homo_taichi_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time7 = time.time()

  time8 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_homo_taichi_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time9 = time.time()

  result2 = jax.block_until_ready(jitconn_matvec_homo_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))

  time12 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_homo_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time13 = time.time()
  # time.sleep(2)

  time14 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_homo_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time15 = time.time()
  # time.sleep(2)

  time16 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_homo_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time17 = time.time()
  # time.sleep(2)

  time18 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_homo_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time19 = time.time()

  time20 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_homo_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time21 = time.time()

  taichi_aot_time1 = (time1 - time0) * 1000
  taichi_aot_time2 = (time3 - time2) * 1000
  taichi_aot_time3 = (time5 - time4) * 1000
  taichi_aot_time4 = (time7 - time6) * 1000
  taichi_aot_time5 = (time9 - time8) * 1000
  brainpy_time1 = (time13 - time12) * 1000
  brainpy_time2 = (time15 - time14) * 1000
  brainpy_time3 = (time17 - time16) * 1000
  brainpy_time4 = (time19 - time18) * 1000
  brainpy_time5 = (time21 - time20) * 1000

  print('taichi_aot_1: ', taichi_aot_time1, 'ms')
  print('taichi_aot_2: ', taichi_aot_time2, 'ms')
  print('taichi_aot_3: ', taichi_aot_time3, 'ms')
  print('taichi_aot_4: ', taichi_aot_time4, 'ms')
  print('taichi_aot_5: ', taichi_aot_time5, 'ms')
  print('brainpylib_cpu_1: ', brainpy_time1, 'ms')
  print('brainpylib_cpu_2: ', brainpy_time2, 'ms')
  print('brainpylib_cpu_3: ', brainpy_time3, 'ms')
  print('brainpylib_cpu_4: ', brainpy_time4, 'ms')
  print('brainpylib_cpu_5: ', brainpy_time5, 'ms')
  # assert(jnp.allclose(result1[0], result2))

  speedup = (brainpy_time1 + brainpy_time2 + brainpy_time3 + brainpy_time4 + brainpy_time5) / \
            (taichi_aot_time1 + taichi_aot_time2 + taichi_aot_time3 + taichi_aot_time4 + taichi_aot_time5) - 1

  return taichi_aot_time1, taichi_aot_time2, taichi_aot_time3, taichi_aot_time4, taichi_aot_time5,\
      brainpy_time1, brainpy_time2, brainpy_time3, brainpy_time4, brainpy_time5, speedup

def test_jitconn_matvec_uniform_cpu(shape, transpose, outdim_parallel):
  rng = bm.random.RandomState(seed=seed)
  events = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

  # groundtruth = bm.as_jax(vector, dtype=float) @ bm.as_jax(dense)

  result1 = jax.block_until_ready(jitconn_matvec_uniform_taichi_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  # time.sleep(2)

  time0 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_uniform_taichi_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time1 = time.time()
  # time.sleep(2)

  time2 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_uniform_taichi_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time3 = time.time()
  # time.sleep(2)

  time4 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_uniform_taichi_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time5 = time.time()
  # time.sleep(2)

  time6 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_uniform_taichi_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time7 = time.time()

  time8 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_uniform_taichi_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time9 = time.time()

  result2 = jax.block_until_ready(jitconn_matvec_uniform_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
#   print(result1[0])
#   print(result2)
#   print(groundtruth - result1[0])
#   print(groundtruth - result2)
  
  # print(result1[0] - result2)
  # print(bm.allclose(groundtruth, result1[0]))
  # print(bm.allclose(groundtruth, result2))
  # assert bm.allclose(result1[0], result2)

  time12 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_uniform_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time13 = time.time()
  # time.sleep(2)

  time14 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_uniform_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time15 = time.time()
  # time.sleep(2)

  time16 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_uniform_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time17 = time.time()
  # time.sleep(2)

  time18 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_uniform_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time19 = time.time()

  time20 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_uniform_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time21 = time.time()

  taichi_aot_time1 = (time1 - time0) * 1000
  taichi_aot_time2 = (time3 - time2) * 1000
  taichi_aot_time3 = (time5 - time4) * 1000
  taichi_aot_time4 = (time7 - time6) * 1000
  taichi_aot_time5 = (time9 - time8) * 1000
  brainpy_time1 = (time13 - time12) * 1000
  brainpy_time2 = (time15 - time14) * 1000
  brainpy_time3 = (time17 - time16) * 1000
  brainpy_time4 = (time19 - time18) * 1000
  brainpy_time5 = (time21 - time20) * 1000

  print('taichi_aot_1: ', taichi_aot_time1, 'ms')
  print('taichi_aot_2: ', taichi_aot_time2, 'ms')
  print('taichi_aot_3: ', taichi_aot_time3, 'ms')
  print('taichi_aot_4: ', taichi_aot_time4, 'ms')
  print('taichi_aot_5: ', taichi_aot_time5, 'ms')
  print('brainpylib_cpu_1: ', brainpy_time1, 'ms')
  print('brainpylib_cpu_2: ', brainpy_time2, 'ms')
  print('brainpylib_cpu_3: ', brainpy_time3, 'ms')
  print('brainpylib_cpu_4: ', brainpy_time4, 'ms')
  print('brainpylib_cpu_5: ', brainpy_time5, 'ms')
  # assert(jnp.allclose(result1[0], result2))

  speedup = (brainpy_time1 + brainpy_time2 + brainpy_time3 + brainpy_time4 + brainpy_time5) / \
            (taichi_aot_time1 + taichi_aot_time2 + taichi_aot_time3 + taichi_aot_time4 + taichi_aot_time5) - 1

  return taichi_aot_time1, taichi_aot_time2, taichi_aot_time3, taichi_aot_time4, taichi_aot_time5,\
      brainpy_time1, brainpy_time2, brainpy_time3, brainpy_time4, brainpy_time5, speedup

def test_jitconn_matvec_normal_cpu(shape, transpose, outdim_parallel):
  rng = bm.random.RandomState(seed=seed)
  events = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

  # groundtruth = bm.as_jax(vector, dtype=float) @ bm.as_jax(dense)

  result1 = jax.block_until_ready(jitconn_matvec_normal_taichi_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  # time.sleep(2)

  time0 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_normal_taichi_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time1 = time.time()
  # time.sleep(2)

  time2 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_normal_taichi_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time3 = time.time()
  # time.sleep(2)

  time4 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_normal_taichi_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time5 = time.time()
  # time.sleep(2)

  time6 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_normal_taichi_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time7 = time.time()

  time8 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_normal_taichi_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time9 = time.time()

  result2 = jax.block_until_ready(jitconn_matvec_normal_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
#   print(result1[0])
#   print(result2)
#   print(groundtruth - result1[0])
#   print(groundtruth - result2)
  
  # print(result1[0] - result2)
  # print(bm.allclose(groundtruth, result1[0]))
  # print(bm.allclose(groundtruth, result2))
  # assert bm.allclose(result1[0], result2)

  time12 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_normal_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time13 = time.time()
  # time.sleep(2)

  time14 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_normal_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time15 = time.time()
  # time.sleep(2)

  time16 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_normal_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time17 = time.time()
  # time.sleep(2)

  time18 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_normal_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time19 = time.time()

  time20 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_normal_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time21 = time.time()

  taichi_aot_time1 = (time1 - time0) * 1000
  taichi_aot_time2 = (time3 - time2) * 1000
  taichi_aot_time3 = (time5 - time4) * 1000
  taichi_aot_time4 = (time7 - time6) * 1000
  taichi_aot_time5 = (time9 - time8) * 1000
  brainpy_time1 = (time13 - time12) * 1000
  brainpy_time2 = (time15 - time14) * 1000
  brainpy_time3 = (time17 - time16) * 1000
  brainpy_time4 = (time19 - time18) * 1000
  brainpy_time5 = (time21 - time20) * 1000

  print('taichi_aot_1: ', taichi_aot_time1, 'ms')
  print('taichi_aot_2: ', taichi_aot_time2, 'ms')
  print('taichi_aot_3: ', taichi_aot_time3, 'ms')
  print('taichi_aot_4: ', taichi_aot_time4, 'ms')
  print('taichi_aot_5: ', taichi_aot_time5, 'ms')
  print('brainpylib_cpu_1: ', brainpy_time1, 'ms')
  print('brainpylib_cpu_2: ', brainpy_time2, 'ms')
  print('brainpylib_cpu_3: ', brainpy_time3, 'ms')
  print('brainpylib_cpu_4: ', brainpy_time4, 'ms')
  print('brainpylib_cpu_5: ', brainpy_time5, 'ms')
  # assert(jnp.allclose(result1[0], result2))

  speedup = (brainpy_time1 + brainpy_time2 + brainpy_time3 + brainpy_time4 + brainpy_time5) / \
            (taichi_aot_time1 + taichi_aot_time2 + taichi_aot_time3 + taichi_aot_time4 + taichi_aot_time5) - 1

  return taichi_aot_time1, taichi_aot_time2, taichi_aot_time3, taichi_aot_time4, taichi_aot_time5,\
      brainpy_time1, brainpy_time2, brainpy_time3, brainpy_time4, brainpy_time5, speedup

def test_jitconn_matvec_homo_gpu(shape, transpose, outdim_parallel):
  rng = bm.random.RandomState(seed=seed)
  vector = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

  # groundtruth = bm.as_jax(vector, dtype=float) @ bm.as_jax(dense)

  result1 = jax.block_until_ready(jitconn_matvec_homo_taichi_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  # time.sleep(2)

  time0 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_homo_taichi_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time1 = time.time()
  # time.sleep(2)

  time2 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_homo_taichi_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time3 = time.time()
  # time.sleep(2)

  time4 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_homo_taichi_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time5 = time.time()
  # time.sleep(2)

  time6 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_homo_taichi_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time7 = time.time()

  time8 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_homo_taichi_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time9 = time.time()

  result2 = jax.block_until_ready(jitconn_matvec_homo_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
#   print(result1[0])
#   print(result2)
#   print(groundtruth - result1[0])
#   print(groundtruth - result2)
  
  # print(result1[0] - result2)
  # print(bm.allclose(groundtruth, result1[0]))
  # print(bm.allclose(groundtruth, result2))
  # assert bm.allclose(result1[0], result2)

  time12 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_homo_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time13 = time.time()
  # time.sleep(2)

  time14 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_homo_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time15 = time.time()
  # time.sleep(2)

  time16 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_homo_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time17 = time.time()
  # time.sleep(2)

  time18 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_homo_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time19 = time.time()

  time20 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_homo_grad(vector, homo_data, conn_prob, seed, shape=shape, outdim_parallel=outdim_parallel, transpose=transpose))
  time21 = time.time()

  taichi_aot_time1 = (time1 - time0) * 1000
  taichi_aot_time2 = (time3 - time2) * 1000
  taichi_aot_time3 = (time5 - time4) * 1000
  taichi_aot_time4 = (time7 - time6) * 1000
  taichi_aot_time5 = (time9 - time8) * 1000
  brainpy_time1 = (time13 - time12) * 1000
  brainpy_time2 = (time15 - time14) * 1000
  brainpy_time3 = (time17 - time16) * 1000
  brainpy_time4 = (time19 - time18) * 1000
  brainpy_time5 = (time21 - time20) * 1000

  print('taichi_aot_1: ', taichi_aot_time1, 'ms')
  print('taichi_aot_2: ', taichi_aot_time2, 'ms')
  print('taichi_aot_3: ', taichi_aot_time3, 'ms')
  print('taichi_aot_4: ', taichi_aot_time4, 'ms')
  print('taichi_aot_5: ', taichi_aot_time5, 'ms')
  print('brainpylib_1: ', brainpy_time1, 'ms')
  print('brainpylib_2: ', brainpy_time2, 'ms')
  print('brainpylib_3: ', brainpy_time3, 'ms')
  print('brainpylib_4: ', brainpy_time4, 'ms')
  print('brainpylib_5: ', brainpy_time5, 'ms')
  # assert(jnp.allclose(result1[0], result2))

  speedup = (brainpy_time1 + brainpy_time2 + brainpy_time3 + brainpy_time4 + brainpy_time5) / \
            (taichi_aot_time1 + taichi_aot_time2 + taichi_aot_time3 + taichi_aot_time4 + taichi_aot_time5) - 1

  return taichi_aot_time1, taichi_aot_time2, taichi_aot_time3, taichi_aot_time4, taichi_aot_time5,\
      brainpy_time1, brainpy_time2, brainpy_time3, brainpy_time4, brainpy_time5, speedup

def test_jitconn_matvec_uniform_gpu(shape, transpose, outdim_parallel):
  rng = bm.random.RandomState(seed=seed)
  events = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

  # groundtruth = bm.as_jax(vector, dtype=float) @ bm.as_jax(dense)

  result1 = jax.block_until_ready(jitconn_matvec_uniform_taichi_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  # time.sleep(2)

  time0 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_uniform_taichi_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time1 = time.time()
  # time.sleep(2)

  time2 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_uniform_taichi_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time3 = time.time()
  # time.sleep(2)

  time4 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_uniform_taichi_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time5 = time.time()
  # time.sleep(2)

  time6 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_uniform_taichi_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time7 = time.time()

  time8 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_uniform_taichi_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time9 = time.time()

  result2 = jax.block_until_ready(jitconn_matvec_uniform_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
#   print(result1[0])
#   print(result2)
#   print(groundtruth - result1[0])
#   print(groundtruth - result2)
  
  # print(result1[0] - result2)
  # print(bm.allclose(groundtruth, result1[0]))
  # print(bm.allclose(groundtruth, result2))
  # assert bm.allclose(result1[0], result2)

  time12 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_uniform_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time13 = time.time()
  # time.sleep(2)

  time14 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_uniform_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time15 = time.time()
  # time.sleep(2)

  time16 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_uniform_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time17 = time.time()
  # time.sleep(2)

  time18 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_uniform_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time19 = time.time()

  time20 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_uniform_grad(events, w_low, w_high, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time21 = time.time()

  taichi_aot_time1 = (time1 - time0) * 1000
  taichi_aot_time2 = (time3 - time2) * 1000
  taichi_aot_time3 = (time5 - time4) * 1000
  taichi_aot_time4 = (time7 - time6) * 1000
  taichi_aot_time5 = (time9 - time8) * 1000
  brainpy_time1 = (time13 - time12) * 1000
  brainpy_time2 = (time15 - time14) * 1000
  brainpy_time3 = (time17 - time16) * 1000
  brainpy_time4 = (time19 - time18) * 1000
  brainpy_time5 = (time21 - time20) * 1000

  print('taichi_aot_1: ', taichi_aot_time1, 'ms')
  print('taichi_aot_2: ', taichi_aot_time2, 'ms')
  print('taichi_aot_3: ', taichi_aot_time3, 'ms')
  print('taichi_aot_4: ', taichi_aot_time4, 'ms')
  print('taichi_aot_5: ', taichi_aot_time5, 'ms')
  print('brainpylib_1: ', brainpy_time1, 'ms')
  print('brainpylib_2: ', brainpy_time2, 'ms')
  print('brainpylib_3: ', brainpy_time3, 'ms')
  print('brainpylib_4: ', brainpy_time4, 'ms')
  print('brainpylib_5: ', brainpy_time5, 'ms')
  # assert(jnp.allclose(result1[0], result2))

  speedup = (brainpy_time1 + brainpy_time2 + brainpy_time3 + brainpy_time4 + brainpy_time5) / \
            (taichi_aot_time1 + taichi_aot_time2 + taichi_aot_time3 + taichi_aot_time4 + taichi_aot_time5) - 1

  return taichi_aot_time1, taichi_aot_time2, taichi_aot_time3, taichi_aot_time4, taichi_aot_time5,\
      brainpy_time1, brainpy_time2, brainpy_time3, brainpy_time4, brainpy_time5, speedup

def test_jitconn_matvec_normal_gpu(shape, transpose, outdim_parallel):
  rng = bm.random.RandomState(seed=seed)
  events = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

  # groundtruth = bm.as_jax(vector, dtype=float) @ bm.as_jax(dense)

  result1 = jax.block_until_ready(jitconn_matvec_normal_taichi_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  # time.sleep(2)

  time0 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_normal_taichi_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time1 = time.time()
  # time.sleep(2)

  time2 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_normal_taichi_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time3 = time.time()
  # time.sleep(2)

  time4 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_normal_taichi_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time5 = time.time()
  # time.sleep(2)

  time6 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_normal_taichi_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time7 = time.time()

  time8 = time.time()
  result1 = jax.block_until_ready(jitconn_matvec_normal_taichi_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time9 = time.time()

  result2 = jax.block_until_ready(jitconn_matvec_normal_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
#   print(result1[0])
#   print(result2)
#   print(groundtruth - result1[0])
#   print(groundtruth - result2)
  
  # print(result1[0] - result2)
  # print(bm.allclose(groundtruth, result1[0]))
  # print(bm.allclose(groundtruth, result2))
  # assert bm.allclose(result1[0], result2)

  time12 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_normal_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time13 = time.time()
  # time.sleep(2)

  time14 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_normal_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time15 = time.time()
  # time.sleep(2)

  time16 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_normal_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time17 = time.time()
  # time.sleep(2)

  time18 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_normal_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time19 = time.time()

  time20 = time.time()
  result2 = jax.block_until_ready(jitconn_matvec_normal_grad(events, w_mu, w_sigma, conn_prob, seed, shape=shape, transpose=transpose, outdim_parallel=outdim_parallel))
  time21 = time.time()

  taichi_aot_time1 = (time1 - time0) * 1000
  taichi_aot_time2 = (time3 - time2) * 1000
  taichi_aot_time3 = (time5 - time4) * 1000
  taichi_aot_time4 = (time7 - time6) * 1000
  taichi_aot_time5 = (time9 - time8) * 1000
  brainpy_time1 = (time13 - time12) * 1000
  brainpy_time2 = (time15 - time14) * 1000
  brainpy_time3 = (time17 - time16) * 1000
  brainpy_time4 = (time19 - time18) * 1000
  brainpy_time5 = (time21 - time20) * 1000

  print('taichi_aot_1: ', taichi_aot_time1, 'ms')
  print('taichi_aot_2: ', taichi_aot_time2, 'ms')
  print('taichi_aot_3: ', taichi_aot_time3, 'ms')
  print('taichi_aot_4: ', taichi_aot_time4, 'ms')
  print('taichi_aot_5: ', taichi_aot_time5, 'ms')
  print('brainpylib_1: ', brainpy_time1, 'ms')
  print('brainpylib_2: ', brainpy_time2, 'ms')
  print('brainpylib_3: ', brainpy_time3, 'ms')
  print('brainpylib_4: ', brainpy_time4, 'ms')
  print('brainpylib_5: ', brainpy_time5, 'ms')
  # assert(jnp.allclose(result1[0], result2))

  speedup = (brainpy_time1 + brainpy_time2 + brainpy_time3 + brainpy_time4 + brainpy_time5) / \
            (taichi_aot_time1 + taichi_aot_time2 + taichi_aot_time3 + taichi_aot_time4 + taichi_aot_time5) - 1

  return taichi_aot_time1, taichi_aot_time2, taichi_aot_time3, taichi_aot_time4, taichi_aot_time5,\
      brainpy_time1, brainpy_time2, brainpy_time3, brainpy_time4, brainpy_time5, speedup


def test_jitconn_matvec_cpu(shape, _type, transpose, outdim_parallel):
  print('shape: ', shape, ' type: ', _type, ' transpose: ', transpose, ' outdim_parallel: ', outdim_parallel)
  if _type == 'homo':
    return test_jitconn_matvec_homo_cpu(shape, transpose, outdim_parallel)
  elif _type == 'uniform':
    return test_jitconn_matvec_uniform_cpu(shape, transpose, outdim_parallel)
  elif _type == 'normal':
    return test_jitconn_matvec_normal_cpu(shape, transpose, outdim_parallel)
  else:
    raise ValueError


def test_jitconn_matvec_gpu(shape, _type, transpose, outdim_parallel):
  print('shape: ', shape, ' type: ', _type, ' transpose: ', transpose, ' outdim_parallel: ', outdim_parallel)
  if _type == 'homo':
    return test_jitconn_matvec_homo_gpu(shape, transpose, outdim_parallel)
  elif _type == 'uniform':
    return test_jitconn_matvec_uniform_gpu(shape, transpose, outdim_parallel)
  elif _type == 'normal':
    return test_jitconn_matvec_normal_gpu(shape, transpose, outdim_parallel)
  else:
    raise ValueError

PATH = os.path.dirname(os.path.abspath(__file__))

# init dataframe
df = pd.DataFrame(columns=['shape[0]', 'shape[1]', 'backend', 'type', 'transpose', 'outdim_parallel',
                           'taichi aot time1(ms)', 'taichi aot time2(ms)', 'taichi aot time3(ms)', 'taichi aot time4(ms)', 'taichi aot time5(ms)',
                           'brainpy time1(ms)', 'brainpy time2(ms)', 'brainpy time3(ms)', 'brainpy time4(ms)', 'brainpy time5(ms)',
                           'speedup'])

### RECTANGULAR MATRIX
if (bm.get_platform() == 'cpu'):
  for shape1 in shape:
    for shape2 in shape:
      for _type in types:
        for _outdim_parallel in outdim_parallel:
          for _transpose in transpose:
            taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5,\
                brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, speedup = test_jitconn_matvec_cpu((shape1, shape2), _type, _transpose, _outdim_parallel)
            # append to dataframe
            df.loc[df.shape[0]] = [shape1, shape2, 'cpu', _type, _transpose, _outdim_parallel,
                                  taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5,
                                  brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, speedup]
  df.to_csv(f'{PATH}/jitconn_matvec_grad_cpu.csv', index=False)

if (bm.get_platform() == 'gpu'):
  for shape1 in shape:
    for shape2 in shape:
      for _type in types:
        for _outdim_parallel in outdim_parallel:
          for _transpose in transpose:
            taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5,\
                brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, speedup = test_jitconn_matvec_gpu((shape1, shape2), _type, _transpose, _outdim_parallel)
            # append to dataframe
            df.loc[df.shape[0]] = [shape1, shape2, 'gpu', _type, _transpose, _outdim_parallel,
                                  taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5,
                                  brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, speedup]
  df.to_csv(f'{PATH}/jitconn_matvec_grad_gpu.csv', index=False)
