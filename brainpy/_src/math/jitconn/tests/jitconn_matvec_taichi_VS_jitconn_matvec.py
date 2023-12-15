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
conn_prob = 0.1
homo_data = 1.
w_low = 0.
w_high = 1.
w_mu = 0.
w_sigma = 0.1

print(bm.get_platform())

def test_jitconn_matvec_homo_cpu(shape, transpose, outdim_parallel):
  rng = bm.random.RandomState(seed=seed)
  vector = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

  # groundtruth = bm.as_jax(vector, dtype=float) @ bm.as_jax(dense)

  result1 = jax.block_until_ready(bm.jitconn.mv_prob_homo_taichi(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  # time.sleep(2)

  time0 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_homo_taichi(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time1 = time.time()
  # time.sleep(2)

  time2 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_homo_taichi(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time3 = time.time()
  # time.sleep(2)

  time4 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_homo_taichi(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time5 = time.time()
  # time.sleep(2)

  time6 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_homo_taichi(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time7 = time.time()

  time8 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_homo_taichi(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time9 = time.time()

  result2 = jax.block_until_ready(bm.jitconn.mv_prob_homo(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
#   print(result1[0])
#   print(result2)
#   print(groundtruth - result1[0])
#   print(groundtruth - result2)
  
  # print(result1[0] - result2)
  # print(bm.allclose(groundtruth, result1[0]))
  # print(bm.allclose(groundtruth, result2))
  # assert bm.allclose(result1[0], result2)

  time12 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_homo(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time13 = time.time()
  # time.sleep(2)

  time14 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_homo(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time15 = time.time()
  # time.sleep(2)

  time16 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_homo(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time17 = time.time()
  # time.sleep(2)

  time18 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_homo(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time19 = time.time()

  time20 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_homo(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
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

  result1 = jax.block_until_ready(bm.jitconn.mv_prob_uniform_taichi(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  # time.sleep(2)

  time0 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_uniform_taichi(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time1 = time.time()
  # time.sleep(2)

  time2 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_uniform_taichi(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time3 = time.time()
  # time.sleep(2)

  time4 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_uniform_taichi(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time5 = time.time()
  # time.sleep(2)

  time6 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_uniform_taichi(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time7 = time.time()

  time8 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_uniform_taichi(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time9 = time.time()

  result2 = jax.block_until_ready(bm.jitconn.mv_prob_uniform(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
#   print(result1[0])
#   print(result2)
#   print(groundtruth - result1[0])
#   print(groundtruth - result2)
  
  # print(result1[0] - result2)
  # print(bm.allclose(groundtruth, result1[0]))
  # print(bm.allclose(groundtruth, result2))
  # assert bm.allclose(result1[0], result2)

  time12 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_uniform(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time13 = time.time()
  # time.sleep(2)

  time14 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_uniform(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time15 = time.time()
  # time.sleep(2)

  time16 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_uniform(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time17 = time.time()
  # time.sleep(2)

  time18 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_uniform(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time19 = time.time()

  time20 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_uniform(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
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

  result1 = jax.block_until_ready(bm.jitconn.mv_prob_normal_taichi(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  # time.sleep(2)

  time0 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_normal_taichi(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time1 = time.time()
  # time.sleep(2)

  time2 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_normal_taichi(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time3 = time.time()
  # time.sleep(2)

  time4 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_normal_taichi(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time5 = time.time()
  # time.sleep(2)

  time6 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_normal_taichi(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time7 = time.time()

  time8 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_normal_taichi(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time9 = time.time()

  result2 = jax.block_until_ready(bm.jitconn.mv_prob_normal(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
#   print(result1[0])
#   print(result2)
#   print(groundtruth - result1[0])
#   print(groundtruth - result2)
  
  # print(result1[0] - result2)
  # print(bm.allclose(groundtruth, result1[0]))
  # print(bm.allclose(groundtruth, result2))
  # assert bm.allclose(result1[0], result2)

  time12 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_normal(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time13 = time.time()
  # time.sleep(2)

  time14 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_normal(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time15 = time.time()
  # time.sleep(2)

  time16 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_normal(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time17 = time.time()
  # time.sleep(2)

  time18 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_normal(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time19 = time.time()

  time20 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_normal(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
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

  result1 = jax.block_until_ready(bm.jitconn.mv_prob_homo_taichi(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  # time.sleep(2)

  time0 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_homo_taichi(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time1 = time.time()
  # time.sleep(2)

  time2 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_homo_taichi(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time3 = time.time()
  # time.sleep(2)

  time4 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_homo_taichi(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time5 = time.time()
  # time.sleep(2)

  time6 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_homo_taichi(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time7 = time.time()

  time8 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_homo_taichi(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time9 = time.time()

  result2 = jax.block_until_ready(bm.jitconn.mv_prob_homo(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
#   print(result1[0])
#   print(result2)
#   print(groundtruth - result1[0])
#   print(groundtruth - result2)
  
  # print(result1[0] - result2)
  # print(bm.allclose(groundtruth, result1[0]))
  # print(bm.allclose(groundtruth, result2))
  # assert bm.allclose(result1[0], result2)

  time12 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_homo(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time13 = time.time()
  # time.sleep(2)

  time14 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_homo(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time15 = time.time()
  # time.sleep(2)

  time16 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_homo(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time17 = time.time()
  # time.sleep(2)

  time18 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_homo(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time19 = time.time()

  time20 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_homo(vector, homo_data, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
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
  print('brainpylib_gpu_1: ', brainpy_time1, 'ms')
  print('brainpylib_gpu_2: ', brainpy_time2, 'ms')
  print('brainpylib_gpu_3: ', brainpy_time3, 'ms')
  print('brainpylib_gpu_4: ', brainpy_time4, 'ms')
  print('brainpylib_gpu_5: ', brainpy_time5, 'ms')
  # assert(jnp.allclose(result1[0], result2))

  speedup = (brainpy_time1 + brainpy_time2 + brainpy_time3 + brainpy_time4 + brainpy_time5) / \
            (taichi_aot_time1 + taichi_aot_time2 + taichi_aot_time3 + taichi_aot_time4 + taichi_aot_time5) - 1

  return taichi_aot_time1, taichi_aot_time2, taichi_aot_time3, taichi_aot_time4, taichi_aot_time5,\
      brainpy_time1, brainpy_time2, brainpy_time3, brainpy_time4, brainpy_time5, speedup

def test_jitconn_matvec_uniform_gpu(shape, transpose, outdim_parallel):
  rng = bm.random.RandomState(seed=seed)
  events = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

  # groundtruth = bm.as_jax(vector, dtype=float) @ bm.as_jax(dense)

  result1 = jax.block_until_ready(bm.jitconn.mv_prob_uniform_taichi(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  # time.sleep(2)

  time0 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_uniform_taichi(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time1 = time.time()
  # time.sleep(2)

  time2 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_uniform_taichi(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time3 = time.time()
  # time.sleep(2)

  time4 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_uniform_taichi(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time5 = time.time()
  # time.sleep(2)

  time6 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_uniform_taichi(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time7 = time.time()

  time8 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_uniform_taichi(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time9 = time.time()

  result2 = jax.block_until_ready(bm.jitconn.mv_prob_uniform(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
#   print(result1[0])
#   print(result2)
#   print(groundtruth - result1[0])
#   print(groundtruth - result2)
  
  # print(result1[0] - result2)
  # print(bm.allclose(groundtruth, result1[0]))
  # print(bm.allclose(groundtruth, result2))
  # assert bm.allclose(result1[0], result2)

  time12 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_uniform(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time13 = time.time()
  # time.sleep(2)

  time14 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_uniform(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time15 = time.time()
  # time.sleep(2)

  time16 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_uniform(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time17 = time.time()
  # time.sleep(2)

  time18 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_uniform(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time19 = time.time()

  time20 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_uniform(events, w_low=w_low, w_high=w_high, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
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
  print('brainpylib_gpu_1: ', brainpy_time1, 'ms')
  print('brainpylib_gpu_2: ', brainpy_time2, 'ms')
  print('brainpylib_gpu_3: ', brainpy_time3, 'ms')
  print('brainpylib_gpu_4: ', brainpy_time4, 'ms')
  print('brainpylib_gpu_5: ', brainpy_time5, 'ms')
  # assert(jnp.allclose(result1[0], result2))

  speedup = (brainpy_time1 + brainpy_time2 + brainpy_time3 + brainpy_time4 + brainpy_time5) / \
            (taichi_aot_time1 + taichi_aot_time2 + taichi_aot_time3 + taichi_aot_time4 + taichi_aot_time5) - 1

  return taichi_aot_time1, taichi_aot_time2, taichi_aot_time3, taichi_aot_time4, taichi_aot_time5,\
      brainpy_time1, brainpy_time2, brainpy_time3, brainpy_time4, brainpy_time5, speedup

def test_jitconn_matvec_normal_gpu(shape, transpose, outdim_parallel):
  rng = bm.random.RandomState(seed=seed)
  events = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))

  # groundtruth = bm.as_jax(vector, dtype=float) @ bm.as_jax(dense)

  result1 = jax.block_until_ready(bm.jitconn.mv_prob_normal_taichi(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  # time.sleep(2)

  time0 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_normal_taichi(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time1 = time.time()
  # time.sleep(2)

  time2 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_normal_taichi(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time3 = time.time()
  # time.sleep(2)

  time4 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_normal_taichi(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time5 = time.time()
  # time.sleep(2)

  time6 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_normal_taichi(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time7 = time.time()

  time8 = time.time()
  result1 = jax.block_until_ready(bm.jitconn.mv_prob_normal_taichi(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time9 = time.time()

  result2 = jax.block_until_ready(bm.jitconn.mv_prob_normal(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
#   print(result1[0])
#   print(result2)
#   print(groundtruth - result1[0])
#   print(groundtruth - result2)
  
  # print(result1[0] - result2)
  # print(bm.allclose(groundtruth, result1[0]))
  # print(bm.allclose(groundtruth, result2))
  # assert bm.allclose(result1[0], result2)

  time12 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_normal(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time13 = time.time()
  # time.sleep(2)

  time14 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_normal(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time15 = time.time()
  # time.sleep(2)

  time16 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_normal(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time17 = time.time()
  # time.sleep(2)

  time18 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_normal(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
  time19 = time.time()

  time20 = time.time()
  result2 = jax.block_until_ready(bm.jitconn.mv_prob_normal(events, w_mu=w_mu, w_sigma=w_sigma, conn_prob=conn_prob, shape=shape, seed=seed, outdim_parallel=outdim_parallel, transpose=transpose))
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
  print('brainpylib_gpu_1: ', brainpy_time1, 'ms')
  print('brainpylib_gpu_2: ', brainpy_time2, 'ms')
  print('brainpylib_gpu_3: ', brainpy_time3, 'ms')
  print('brainpylib_gpu_4: ', brainpy_time4, 'ms')
  print('brainpylib_gpu_5: ', brainpy_time5, 'ms')
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
  df.to_csv(f'{PATH}/jitconn_matvec_cpu.csv', index=False)

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
  df.to_csv(f'{PATH}/jitconn_matvec_gpu.csv', index=False)

# if (bm.get_platform() == 'gpu'):
#   for _s in s:
#     for _p in p:
#       taichi_aot_avg_time = test_event_ell_gpu_taichi(_s, _p)
#       df.loc[df.shape[0]] = [_s, _p, 'gpu', block_dim, taichi_aot_avg_time, 0]
#   df.to_csv('event_ell_gpu.csv', index=False)

  # df = pd.read_csv('event_ell_gpu.csv')
  # for _s in s:
  #     for _p in p:
  #         brainpy_avg_time = test_event_ell_gpu_brainpylib(_s, _p)
  #         # 找到对应的行
  #         df.loc[(df['s'] == _s) & (df['p'] == _p) & (df['backend'] == 'gpu'), 'brainpy avg time(ms)'] = brainpy_avg_time
  # df.to_csv('event_ell_gpu.csv', index=False)
