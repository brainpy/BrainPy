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

s = [1000, 2500, 5000, 10000, 25000, 50000]
p = [0.1, 0.2, 0.3, 0.4, 0.5]
values_type = ['homo', 'heter']
events_type = ['bool', 'float']

print(bm.get_platform())


def test_event_ell_cpu(s, p, values_type, events_type):
  print('s: ', s, 'p: ', p)
  k = int(s * p)
  bm.random.seed(1234)
  rng = bm.random.RandomState(seed=1234)
  # init
  indices = bm.random.randint(0, s, (s, k))
  vector = bm.random.rand(s) < 0.5
  weight = jnp.array([1.0])
  csr_indices = indices.flatten()
  csr_indptr = np.cumsum(np.insert(np.ones(s, dtype=int) * k, 0, 0))

  pre_indices = np.repeat(np.arange(s), k)
  dense = np.zeros((s, s))
  dense[pre_indices, csr_indices] = 1.0

  if events_type == 'float':
    vector = vector.astype(np.float32)
    vector[vector == 1.0] = bm.random.rand(bm.sum(vector == 1.0))
  if values_type == 'heter':
    heter_data = bm.as_jax(rng.random(csr_indices.shape))
    weight = heter_data

  # groundtruth = bm.as_jax(vector, dtype=float) @ bm.as_jax(dense)

  result1 = jax.block_until_ready(bm.event.csrmv_taichi(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  # time.sleep(2)

  time0 = time.time()
  result1 = jax.block_until_ready(bm.event.csrmv_taichi(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time1 = time.time()
  # time.sleep(2)

  time2 = time.time()
  result1 = jax.block_until_ready(bm.event.csrmv_taichi(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time3 = time.time()
  # time.sleep(2)

  time4 = time.time()
  result1 = jax.block_until_ready(bm.event.csrmv_taichi(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time5 = time.time()
  # time.sleep(2)

  time6 = time.time()
  result1 = jax.block_until_ready(bm.event.csrmv_taichi(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time7 = time.time()

  time8 = time.time()
  result1 = jax.block_until_ready(bm.event.csrmv_taichi(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time9 = time.time()

  result2 = jax.block_until_ready(bm.event.csrmv(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
#   print(result1[0])
#   print(result2)
#   print(groundtruth - result1[0])
#   print(groundtruth - result2)
  
  # print(result1[0] - result2)
  # print(bm.allclose(groundtruth, result1[0]))
  # print(bm.allclose(groundtruth, result2))
  # assert bm.allclose(result1[0], result2)

  time12 = time.time()
  result2 = jax.block_until_ready(bm.event.csrmv(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time13 = time.time()
  # time.sleep(2)

  time14 = time.time()
  result2 = jax.block_until_ready(bm.event.csrmv(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time15 = time.time()
  # time.sleep(2)

  time16 = time.time()
  result2 = jax.block_until_ready(bm.event.csrmv(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time17 = time.time()
  # time.sleep(2)

  time18 = time.time()
  result2 = jax.block_until_ready(bm.event.csrmv(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time19 = time.time()

  time20 = time.time()
  result2 = jax.block_until_ready(bm.event.csrmv(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
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
  assert(jnp.allclose(result1[0], result2))

  speedup = (brainpy_time1 + brainpy_time2 + brainpy_time3 + brainpy_time4 + brainpy_time5) / \
            (taichi_aot_time1 + taichi_aot_time2 + taichi_aot_time3 + taichi_aot_time4 + taichi_aot_time5) - 1

  return taichi_aot_time1, taichi_aot_time2, taichi_aot_time3, taichi_aot_time4, taichi_aot_time5,\
      brainpy_time1, brainpy_time2, brainpy_time3, brainpy_time4, brainpy_time5, speedup

def test_event_ell_gpu(s, p, values_type, events_type):
  print('s: ', s, 'p: ', p)
  k = int(s * p)
  bm.random.seed(1234)
  rng = bm.random.RandomState(seed=1234)
  # init
  indices = bm.random.randint(0, s, (s, k))
  vector = bm.random.rand(s) < 0.5
  weight = jnp.array([1.0])
  csr_indices = indices.flatten()
  csr_indptr = np.cumsum(np.insert(np.ones(s, dtype=int) * k, 0, 0))
  pre_indices = np.repeat(np.arange(s), k)
  dense = np.zeros((s, s))
  dense[pre_indices, csr_indices] = 1.0

  if events_type == 'float':
    vector = vector.astype(np.float32)
    vector[vector == 1.0] = bm.random.rand(bm.sum(vector == 1.0))
  if values_type == 'heter':
    heter_data = bm.as_jax(rng.random(csr_indices.shape))
    weight = heter_data

  groundtruth = bm.as_jax(vector, dtype=float) @ bm.as_jax(dense)



  result1 = jax.block_until_ready(bm.event.csrmv_taichi(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  # time.sleep(2)

  time0 = time.time()
  result1 = jax.block_until_ready(bm.event.csrmv_taichi(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time1 = time.time()
  # time.sleep(2)

  time2 = time.time()
  result1 = jax.block_until_ready(bm.event.csrmv_taichi(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time3 = time.time()
  # time.sleep(2)

  time4 = time.time()
  result1 = jax.block_until_ready(bm.event.csrmv_taichi(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time5 = time.time()
  # time.sleep(2)

  time6 = time.time()
  result1 = jax.block_until_ready(bm.event.csrmv_taichi(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time7 = time.time()

  time8 = time.time()
  result1 = jax.block_until_ready(bm.event.csrmv_taichi(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time9 = time.time()

  result2 = jax.block_until_ready(bm.event.csrmv(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  # print('--------------------result1[0]------------------')
  # print(result1[0])
  # print('--------------------result2------------------')
  # print(result2)
  # print('--------------------gt - result1[0]------------------')
  # print(groundtruth - result1[0])
  # print('--------------------gt - result2------------------')
  # print(groundtruth - result2)
  
  # print(result1[0] - result2)
  # print(bm.allclose(groundtruth, result1[0]))
  # print(bm.allclose(groundtruth, result2))
  # assert bm.allclose(result1[0], result2)

  time12 = time.time()
  result2 = jax.block_until_ready(bm.event.csrmv(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time13 = time.time()
  # time.sleep(2)

  time14 = time.time()
  result2 = jax.block_until_ready(bm.event.csrmv(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time15 = time.time()
  # time.sleep(2)

  time16 = time.time()
  result2 = jax.block_until_ready(bm.event.csrmv(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time17 = time.time()
  # time.sleep(2)

  time18 = time.time()
  result2 = jax.block_until_ready(bm.event.csrmv(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
  time19 = time.time()

  time20 = time.time()
  result2 = jax.block_until_ready(bm.event.csrmv(weight, csr_indices, csr_indptr, vector, shape=(s, s), transpose=True))
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

bm.set_platform('cpu')
block_dim = 64
# init dataframe
df = pd.DataFrame(columns=['s', 'p', 'backend', 'values type', 'events type',
                           'taichi aot time1(ms)', 'taichi aot time2(ms)', 'taichi aot time3(ms)', 'taichi aot time4(ms)', 'taichi aot time5(ms)',
                           'brainpy time1(ms)', 'brainpy time2(ms)', 'brainpy time3(ms)', 'brainpy time4(ms)', 'brainpy time5(ms)',
                           'speedup'])

if (bm.get_platform() == 'cpu'):
  for _s in s:
      for _p in p:
          for _values_type in values_type:
             for _events_type in events_type:
              taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5,\
                  brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, speedup = test_event_ell_cpu(_s, _p, _values_type, _events_type)
              # append to dataframe
              df.loc[df.shape[0]] = [_s, _p, 'cpu', _values_type, _events_type,
                                    taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5,
                                    brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, speedup]
  df.to_csv('event_csrmv_cpu.csv', index=False)

if (bm.get_platform() == 'gpu'):
  for _s in s:
      for _p in p:
          for _values_type in values_type:
             for _events_type in events_type:
              taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5,\
                  brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, speedup = test_event_ell_gpu(_s, _p, _values_type, _events_type)
              # append to dataframe
              df.loc[df.shape[0]] = [_s, _p, 'gpu', _values_type, _events_type,
                                    taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5,
                                    brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, speedup]
  df.to_csv('event_csrmv_gpu.csv', index=False)

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
