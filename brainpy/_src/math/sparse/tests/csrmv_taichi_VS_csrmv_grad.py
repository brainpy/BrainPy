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

s = [1000, 
     5000, 
     10000, 
     15000, 
     20000, 
     25000, 
     30000]
p = [0.1, 0.2, 0.3, 0.4, 0.5]

shape = [
        1000, 
        2500, 
        5000, 
        10000, 
        25000, 
        37500, 
        50000
]

values_type = [
               'homo', 
               'heter'
              ]
events_type = ['float']
transpose = [
             True, 
             False
             ]
method = 'cusparse'

ITERATION = 100
if bm.get_platform() == 'cpu':
  ITERATION = 10

print(bm.get_platform())

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

@partial(jax.jit, static_argnums=(4, 5))
def csrmv_taichi_grad(weight, indices, indptr, vector, shape, transpose):
  r = 0
  for i in range(ITERATION):
    r += jax.grad(sum_op2(bm.sparse.csrmv_taichi), argnums=3)(
      weight, indices, indptr, vector.astype(float), shape=shape, transpose=transpose)
  return r

@partial(jax.jit, static_argnums=(4, 5))
def csrmv_grad(weight, indices, indptr, vector, shape, transpose):
  r = 0
  for i in range(ITERATION):
    r += jax.grad(sum_op(bm.sparse.csrmv), argnums=3)(
    weight, indices, indptr, vector.astype(float), shape=shape, transpose=transpose)
  return r

def test_sparse_csrmv(shape, values_type, events_type, transpose):
  rng = bm.random.RandomState(seed=1234)
  indices, indptr = bp.conn.FixedProb(0.05, seed=1234, allow_multi_conn=True)(*shape).require('pre2post')
  vector = rng.random(shape[0] if transpose else shape[1]) < 0.1
  weight = 1.
  

  if events_type == 'float':
    vector = vector.astype(bm.float32)
  if values_type == 'heter':
    heter_data = bm.ones(indices.shape) * weight
    weight = heter_data

  result = jax.block_until_ready(csrmv_taichi_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  result = jax.block_until_ready(csrmv_taichi_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  result = jax.block_until_ready(csrmv_taichi_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  result = jax.block_until_ready(csrmv_taichi_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  result = jax.block_until_ready(csrmv_taichi_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))

  time0 = time.time()
  result = jax.block_until_ready(csrmv_taichi_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time1 = time.time()

  time2 = time.time()
  result = jax.block_until_ready(csrmv_taichi_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time3 = time.time()

  time4 = time.time()
  result = jax.block_until_ready(csrmv_taichi_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time5 = time.time()

  time6 = time.time()
  result = jax.block_until_ready(csrmv_taichi_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time7 = time.time()

  time8 = time.time()
  result = jax.block_until_ready(csrmv_taichi_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time9 = time.time()
  
  time10 = time.time()
  result = jax.block_until_ready(csrmv_taichi_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time11 = time.time()
  
  time12 = time.time()
  result = jax.block_until_ready(csrmv_taichi_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time13 = time.time()
  
  time14 = time.time()
  result = jax.block_until_ready(csrmv_taichi_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time15 = time.time()
  
  time16 = time.time()
  result = jax.block_until_ready(csrmv_taichi_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time17 = time.time()
  
  time18 = time.time()
  result = jax.block_until_ready(csrmv_taichi_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time19 = time.time()
  

  result = jax.block_until_ready(csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  result = jax.block_until_ready(csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  result = jax.block_until_ready(csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  result = jax.block_until_ready(csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  result = jax.block_until_ready(csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))

  time20 = time.time()
  result = jax.block_until_ready(csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time21 = time.time()

  time22 = time.time()
  result = jax.block_until_ready(csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time23 = time.time()

  time24 = time.time()
  result = jax.block_until_ready(csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time25 = time.time()

  time26 = time.time()
  result = jax.block_until_ready(csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time27 = time.time()

  time28 = time.time()
  result = jax.block_until_ready(csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time29 = time.time()
  
  time30 = time.time()
  result = jax.block_until_ready(csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time31 = time.time()
  
  time32 = time.time()
  result = jax.block_until_ready(csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time33 = time.time()
  
  time34 = time.time()
  result = jax.block_until_ready(csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time35 = time.time()
  
  time36 = time.time()
  result = jax.block_until_ready(csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time37 = time.time()
  
  time38 = time.time()
  result = jax.block_until_ready(csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
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
  print('shape: ', shape, 'values_type: ', values_type, 'events_type: ', events_type, 'transpose: ', transpose)
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

PATH = os.path.dirname(os.path.abspath(__file__))

# init dataframe
df = pd.DataFrame(columns=['s', 'p', 'shape[0]', 'shape[1]', 'backend', 'values type', 'events type', 'transpose',
                           'taichi aot time1(ms)', 'taichi aot time2(ms)', 'taichi aot time3(ms)', 'taichi aot time4(ms)', 'taichi aot time5(ms)',
                           'taichi aot time6(ms)', 'taichi aot time7(ms)', 'taichi aot time8(ms)', 'taichi aot time9(ms)', 'taichi aot time10(ms)',
                           'brainpy time1(ms)', 'brainpy time2(ms)', 'brainpy time3(ms)', 'brainpy time4(ms)', 'brainpy time5(ms)',
                           'brainpy time6(ms)', 'brainpy time7(ms)', 'brainpy time8(ms)', 'brainpy time9(ms)', 'brainpy time10(ms)'])


### RECTANGULAR MATRIX
if (bm.get_platform() == 'cpu'):
  for shape1 in shape:
    for shape2 in shape:
      for _values_type in values_type:
          for _events_type in events_type:
            for _transpose in transpose:
              taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5,\
                taichi_aot_time_6, taichi_aot_time_7, taichi_aot_time_8, taichi_aot_time_9, taichi_aot_time_10,\
                  brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, \
                  brainpy_time_6, brainpy_time_7, brainpy_time_8, brainpy_time_9, brainpy_time_10 = test_sparse_csrmv((shape1, shape2), _values_type, _events_type, _transpose)
              # append to dataframe
              df.loc[df.shape[0]] = [(shape1, shape2), 0.5 , shape1, shape2, 'cpu', _values_type, _events_type, _transpose,
                                    taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5,
                                    taichi_aot_time_6, taichi_aot_time_7, taichi_aot_time_8, taichi_aot_time_9, taichi_aot_time_10,
                                    brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, 
                                    brainpy_time_6, brainpy_time_7, brainpy_time_8, brainpy_time_9, brainpy_time_10]
  df.to_csv(f'{PATH}/csrmv_grad_cpu.csv', index=False)

if (bm.get_platform() == 'gpu'):
  for shape1 in shape:
    for shape2 in shape:
      for _values_type in values_type:
          for _events_type in events_type:
            for _transpose in transpose:
              taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5,\
                taichi_aot_time_6, taichi_aot_time_7, taichi_aot_time_8, taichi_aot_time_9, taichi_aot_time_10,\
                  brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, \
                  brainpy_time_6, brainpy_time_7, brainpy_time_8, brainpy_time_9, brainpy_time_10 = test_sparse_csrmv((shape1, shape2), _values_type, _events_type, _transpose)
              # append to dataframe
              df.loc[df.shape[0]] = [(shape1, shape2), 0.5 , shape1, shape2, 'gpu', _values_type, _events_type, _transpose,
                                    taichi_aot_time_1, taichi_aot_time_2, taichi_aot_time_3, taichi_aot_time_4, taichi_aot_time_5,
                                    taichi_aot_time_6, taichi_aot_time_7, taichi_aot_time_8, taichi_aot_time_9, taichi_aot_time_10,
                                    brainpy_time_1, brainpy_time_2, brainpy_time_3, brainpy_time_4, brainpy_time_5, 
                                    brainpy_time_6, brainpy_time_7, brainpy_time_8, brainpy_time_9, brainpy_time_10]
  df.to_csv(f'{PATH}/csrmv_grad_gpu.csv', index=False)
