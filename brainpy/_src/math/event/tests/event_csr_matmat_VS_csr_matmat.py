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

size = [
  (100, 100, 100),
  (100, 1000, 100),
  (1000, 1000, 100),
  (1000, 1000, 1000),
  (100, 10000, 100),
  (10000, 100, 1000),
  (1000, 100, 10000),
  (10000, 10000, 1000),
  (10000, 1000, 10000),
  (10000, 10000, 10000),
  (20000, 20000, 20000),
]

values_type = [
  'heter',
  'homo',
]
events_type = ['bool',
               'float',
               ]
transpose = [
  # True,
  False
]

ITERATION = 100
if bm.get_platform() == 'cpu':
  ITERATION = 10

print(bm.get_platform())


@partial(jax.jit, static_argnums=(4, 5))
def csrmm(weight, indices, indptr, matrix, shape, transpose):
  r = 0
  for i in range(ITERATION):
    r += bm.sparse.csrmm(weight, indices, indptr, matrix, shape=shape, transpose=transpose)
  return r


@partial(jax.jit, static_argnums=(4, 5))
def event_csrmm(weight, indices, indptr, matrix, shape, transpose):
  r = 0
  for i in range(ITERATION):
    r += bm.event.csrmm(weight, indices, indptr, matrix, shape=shape, transpose=transpose)
  return r


def test_sparse_csrmm(shape, values_type, events_type, transpose):
  rng = bm.random.RandomState(seed=1234)
  matrix1_shape = (shape[1], shape[0]) if transpose else (shape[0], shape[1])
  matrix2_shape = (shape[1], shape[2])
  indices, indptr = bp.conn.FixedProb(0.05, seed=1234, allow_multi_conn=True)(*matrix1_shape).require('pre2post')
  matrix = rng.random(matrix2_shape)
  matrix = bm.as_jax(matrix)
  weight = 1.

  if events_type == 'float':
    matrix = matrix.astype(bm.float32)
  if values_type == 'heter':
    heter_data = bm.ones(indices.shape) * weight
    weight = heter_data

  result = jax.block_until_ready(csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  result = jax.block_until_ready(csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  result = jax.block_until_ready(csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  result = jax.block_until_ready(csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  result = jax.block_until_ready(csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))

  time0 = time.time()
  result = jax.block_until_ready(csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time1 = time.time()

  time2 = time.time()
  result = jax.block_until_ready(csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time3 = time.time()

  time4 = time.time()
  result = jax.block_until_ready(csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time5 = time.time()

  time6 = time.time()
  result = jax.block_until_ready(csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time7 = time.time()

  time8 = time.time()
  result = jax.block_until_ready(csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time9 = time.time()

  time10 = time.time()
  result = jax.block_until_ready(csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time11 = time.time()

  time12 = time.time()
  result = jax.block_until_ready(csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time13 = time.time()

  time14 = time.time()
  result = jax.block_until_ready(csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time15 = time.time()

  time16 = time.time()
  result = jax.block_until_ready(csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time17 = time.time()

  time18 = time.time()
  result = jax.block_until_ready(csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time19 = time.time()

  result1 = result

  result = jax.block_until_ready(event_csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  result = jax.block_until_ready(event_csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  result = jax.block_until_ready(event_csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  result = jax.block_until_ready(event_csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  result = jax.block_until_ready(event_csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))

  time20 = time.time()
  result = jax.block_until_ready(event_csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time21 = time.time()

  result2 = result

  time22 = time.time()
  result = jax.block_until_ready(event_csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time23 = time.time()

  time24 = time.time()
  result = jax.block_until_ready(event_csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time25 = time.time()

  time26 = time.time()
  result = jax.block_until_ready(event_csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time27 = time.time()

  time28 = time.time()
  result = jax.block_until_ready(event_csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time29 = time.time()

  time30 = time.time()
  result = jax.block_until_ready(event_csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time31 = time.time()

  time32 = time.time()
  result = jax.block_until_ready(event_csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time33 = time.time()

  time34 = time.time()
  result = jax.block_until_ready(event_csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time35 = time.time()

  time36 = time.time()
  result = jax.block_until_ready(event_csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time37 = time.time()

  time38 = time.time()
  result = jax.block_until_ready(event_csrmm(weight, indices, indptr, matrix, shape=matrix1_shape, transpose=transpose))
  time39 = time.time()

  csrmm_time1 = (time1 - time0) * 1000
  csrmm_time2 = (time3 - time2) * 1000
  csrmm_time3 = (time5 - time4) * 1000
  csrmm_time4 = (time7 - time6) * 1000
  csrmm_time5 = (time9 - time8) * 1000
  csrmm_time6 = (time11 - time10) * 1000
  csrmm_time7 = (time13 - time12) * 1000
  csrmm_time8 = (time15 - time14) * 1000
  csrmm_time9 = (time17 - time16) * 1000
  csrmm_time10 = (time19 - time18) * 1000
  event_csrmm_time1 = (time21 - time20) * 1000
  event_csrmm_time2 = (time23 - time22) * 1000
  event_csrmm_time3 = (time25 - time24) * 1000
  event_csrmm_time4 = (time27 - time26) * 1000
  event_csrmm_time5 = (time29 - time28) * 1000
  event_csrmm_time6 = (time31 - time30) * 1000
  event_csrmm_time7 = (time33 - time32) * 1000
  event_csrmm_time8 = (time35 - time34) * 1000
  event_csrmm_time9 = (time37 - time36) * 1000
  event_csrmm_time10 = (time39 - time38) * 1000
  print('shape: ', shape, 'values_type: ', values_type, 'events_type: ', events_type, 'transpose: ', transpose)
  print('csrmm_1: ', csrmm_time1, 'ms')
  print('csrmm_3: ', csrmm_time3, 'ms')
  print('csrmm_5: ', csrmm_time5, 'ms')
  print('csrmm_7: ', csrmm_time7, 'ms')
  print('csrmm_9: ', csrmm_time9, 'ms')
  print('event_csrmm_1: ', event_csrmm_time1, 'ms')
  print('event_csrmm_3: ', event_csrmm_time3, 'ms')
  print('event_csrmm_5: ', event_csrmm_time5, 'ms')
  print('event_csrmm_7: ', event_csrmm_time7, 'ms')
  print('event_csrmm_9: ', event_csrmm_time9, 'ms')

  r = bm.allclose(result1, result2)
  if not r:
    print('result1: ', result1)
    print('result2: ', result2)

  return csrmm_time1, csrmm_time2, csrmm_time3, csrmm_time4, csrmm_time5, \
    csrmm_time6, csrmm_time7, csrmm_time8, csrmm_time9, csrmm_time10, \
    event_csrmm_time1, event_csrmm_time2, event_csrmm_time3, event_csrmm_time4, event_csrmm_time5, \
    event_csrmm_time6, event_csrmm_time7, event_csrmm_time8, event_csrmm_time9, event_csrmm_time10


PATH = os.path.dirname(os.path.abspath(__file__))

# init dataframe
df = pd.DataFrame(
  columns=['shape', 'p', 'shape[0]', 'shape[1]', 'shape[2]', 'backend', 'values type', 'events type', 'transpose',
           'csrmm time1(ms)', 'csrmm time2(ms)', 'csrmm time3(ms)', 'csrmm time4(ms)',
           'csrmm time5(ms)',
           'csrmm time6(ms)', 'csrmm time7(ms)', 'csrmm time8(ms)', 'csrmm time9(ms)',
           'csrmm time10(ms)',
           'event_csrmm time1(ms)', 'event_csrmm time2(ms)', 'event_csrmm time3(ms)', 'event_csrmm time4(ms)',
           'event_csrmm time5(ms)',
           'event_csrmm time6(ms)', 'event_csrmm time7(ms)', 'event_csrmm time8(ms)', 'event_csrmm time9(ms)',
           'event_csrmm time10(ms)'])

### RECTANGULAR MATRIX
if (bm.get_platform() == 'cpu'):
  for shape in size:
    for _values_type in values_type:
      for _events_type in events_type:
        for _transpose in transpose:
          csrmm_time_1, csrmm_time_2, csrmm_time_3, csrmm_time_4, csrmm_time_5, \
            csrmm_time_6, csrmm_time_7, csrmm_time_8, csrmm_time_9, csrmm_time_10, \
            event_csrmm_time_1, event_csrmm_time_2, event_csrmm_time_3, event_csrmm_time_4, event_csrmm_time_5, \
            event_csrmm_time_6, event_csrmm_time_7, event_csrmm_time_8, event_csrmm_time_9, event_csrmm_time_10 = test_sparse_csrmm(
            shape,
            _values_type,
            _events_type,
            _transpose)
          # append to dataframe
          df.loc[df.shape[0]] = [shape, 0.05, shape[0], shape[1], shape[2], 'cpu', _values_type, _events_type,
                                 _transpose,
                                 csrmm_time_1, csrmm_time_2, csrmm_time_3, csrmm_time_4,
                                 csrmm_time_5,
                                 csrmm_time_6, csrmm_time_7, csrmm_time_8, csrmm_time_9,
                                 csrmm_time_10,
                                 event_csrmm_time_1, event_csrmm_time_2, event_csrmm_time_3, event_csrmm_time_4,
                                 event_csrmm_time_5,
                                 event_csrmm_time_6, event_csrmm_time_7, event_csrmm_time_8, event_csrmm_time_9,
                                 event_csrmm_time_10]
          df.to_csv(f'{PATH}/csrmm_cpu.csv', index=False)

if (bm.get_platform() == 'gpu'):
  for shape in size:
    for _values_type in values_type:
      for _events_type in events_type:
        for _transpose in transpose:
          csrmm_time_1, csrmm_time_2, csrmm_time_3, csrmm_time_4, csrmm_time_5, \
            csrmm_time_6, csrmm_time_7, csrmm_time_8, csrmm_time_9, csrmm_time_10, \
            event_csrmm_time_1, event_csrmm_time_2, event_csrmm_time_3, event_csrmm_time_4, event_csrmm_time_5, \
            event_csrmm_time_6, event_csrmm_time_7, event_csrmm_time_8, event_csrmm_time_9, event_csrmm_time_10 = test_sparse_csrmm(
            shape,
            _values_type,
            _events_type,
            _transpose)
          # append to dataframe
          df.loc[df.shape[0]] = [shape, 0.05, shape[0], shape[1], shape[2], 'gpu', _values_type, _events_type,
                                 _transpose,
                                 csrmm_time_1, csrmm_time_2, csrmm_time_3, csrmm_time_4,
                                 csrmm_time_5,
                                 csrmm_time_6, csrmm_time_7, csrmm_time_8, csrmm_time_9,
                                 csrmm_time_10,
                                 event_csrmm_time_1, event_csrmm_time_2, event_csrmm_time_3, event_csrmm_time_4,
                                 event_csrmm_time_5,
                                 event_csrmm_time_6, event_csrmm_time_7, event_csrmm_time_8, event_csrmm_time_9,
                                 event_csrmm_time_10]
          df.to_csv(f'{PATH}/csrmm_gpu.csv', index=False)
