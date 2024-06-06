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

s = [1000, 5000, 10000, 20000, 25000]
p = 0.05

shape = [
  1000,
  2500,
  5000,
  10000,
  15000,
  20000,
  25000
]

values_type = [
  # 'homo',
  'heter',
]
events_type = [
  'bool',
  'float',
]
transpose = [
  True,
  False,
]
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
def event_csrmv_grad(weight, indices, indptr, vector, shape, transpose):
  r = 0
  for i in range(ITERATION):
    r += jax.grad(sum_op2(bm.event.csrmv), argnums=0)(
      weight, indices, indptr, vector.astype(float), shape=shape, transpose=transpose)
  return r


@partial(jax.jit, static_argnums=(4, 5))
def normal_csrmv_grad(weight, indices, indptr, vector, shape, transpose):
  r = 0
  for i in range(ITERATION):
    r += jax.grad(sum_op2(bm.sparse.csrmv), argnums=0)(
      weight, indices, indptr, vector.astype(float), shape=shape, transpose=transpose)
  return r


def test_event_csrmv_dW(shape, values_type, events_type, transpose):
  rng = bm.random.RandomState(1234)
  indices, indptr = bp.conn.FixedProb(p, seed=1234, allow_multi_conn=True)(*shape).require('pre2post')
  vector = rng.random(shape[0] if transpose else shape[1]) < 0.1
  weight = 1.

  if events_type == 'float':
    vector = vector.astype(bm.float32)
  if values_type == 'heter':
    heter_data = bm.ones(indices.shape) * weight
    weight = heter_data

  result = jax.block_until_ready(
    event_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  result = jax.block_until_ready(
    event_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  result = jax.block_until_ready(
    event_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  result = jax.block_until_ready(
    event_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  result = jax.block_until_ready(
    event_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))

  time0 = time.time()
  result = jax.block_until_ready(
    event_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time1 = time.time()

  time2 = time.time()
  result = jax.block_until_ready(
    event_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time3 = time.time()

  time4 = time.time()
  result = jax.block_until_ready(
    event_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time5 = time.time()

  time6 = time.time()
  result = jax.block_until_ready(
    event_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time7 = time.time()

  time8 = time.time()
  result = jax.block_until_ready(
    event_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time9 = time.time()

  time10 = time.time()
  result = jax.block_until_ready(
    event_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time11 = time.time()

  time12 = time.time()
  result = jax.block_until_ready(
    event_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time13 = time.time()

  time14 = time.time()
  result = jax.block_until_ready(
    event_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time15 = time.time()

  time16 = time.time()
  result = jax.block_until_ready(
    event_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time17 = time.time()

  time18 = time.time()
  result = jax.block_until_ready(
    event_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time19 = time.time()

  result = jax.block_until_ready(normal_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  result = jax.block_until_ready(normal_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  result = jax.block_until_ready(normal_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  result = jax.block_until_ready(normal_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  result = jax.block_until_ready(normal_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))

  time20 = time.time()
  result = jax.block_until_ready(normal_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time21 = time.time()

  time22 = time.time()
  result = jax.block_until_ready(normal_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time23 = time.time()

  time24 = time.time()
  result = jax.block_until_ready(normal_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time25 = time.time()

  time26 = time.time()
  result = jax.block_until_ready(normal_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time27 = time.time()

  time28 = time.time()
  result = jax.block_until_ready(normal_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time29 = time.time()

  time30 = time.time()
  result = jax.block_until_ready(normal_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time31 = time.time()

  time32 = time.time()
  result = jax.block_until_ready(normal_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time33 = time.time()

  time34 = time.time()
  result = jax.block_until_ready(normal_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time35 = time.time()

  time36 = time.time()
  result = jax.block_until_ready(normal_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time37 = time.time()

  time38 = time.time()
  result = jax.block_until_ready(normal_csrmv_grad(weight, indices, indptr, vector, shape=shape, transpose=transpose))
  time39 = time.time()

  event_time1 = (time1 - time0) * 1000
  event_time2 = (time3 - time2) * 1000
  event_time3 = (time5 - time4) * 1000
  event_time4 = (time7 - time6) * 1000
  event_time5 = (time9 - time8) * 1000
  event_time6 = (time11 - time10) * 1000
  event_time7 = (time13 - time12) * 1000
  event_time8 = (time15 - time14) * 1000
  event_time9 = (time17 - time16) * 1000
  event_time10 = (time19 - time18) * 1000
  normal_time1 = (time21 - time20) * 1000
  normal_time2 = (time23 - time22) * 1000
  normal_time3 = (time25 - time24) * 1000
  normal_time4 = (time27 - time26) * 1000
  normal_time5 = (time29 - time28) * 1000
  normal_time6 = (time31 - time30) * 1000
  normal_time7 = (time33 - time32) * 1000
  normal_time8 = (time35 - time34) * 1000
  normal_time9 = (time37 - time36) * 1000
  normal_time10 = (time39 - time38) * 1000
  print('shape: ', shape, 'values_type: ', values_type, 'events_type: ', events_type, 'transpose: ', transpose)
  print('event_time1: ', event_time1, 'ms')
  print('event_time3: ', event_time3, 'ms')
  print('event_time5: ', event_time5, 'ms')
  print('event_time7: ', event_time7, 'ms')
  print('event_time9: ', event_time9, 'ms')
  print('normal_time1: ', normal_time1, 'ms')
  print('normal_time3: ', normal_time3, 'ms')
  print('normal_time5: ', normal_time5, 'ms')
  print('normal_time7: ', normal_time7, 'ms')
  print('normal_time9: ', normal_time9, 'ms')

  return event_time1, event_time2, event_time3, event_time4, event_time5, \
    event_time6, event_time7, event_time8, event_time9, event_time10, \
    normal_time1, normal_time2, normal_time3, normal_time4, normal_time5, \
    normal_time6, normal_time7, normal_time8, normal_time9, normal_time10


PATH = os.path.dirname(os.path.abspath(__file__))

# init dataframe
df = pd.DataFrame(columns=['s', 'p', 'shape[0]', 'shape[1]', 'backend', 'values type', 'events type', 'transpose',
                           'event time1(ms)', 'event time2(ms)', 'event time3(ms)', 'event time4(ms)',
                           'event time5(ms)',
                           'event time6(ms)', 'event time7(ms)', 'event time8(ms)', 'event time9(ms)',
                           'event time10(ms)',
                           'normal time1(ms)', 'normal time2(ms)', 'normal time3(ms)', 'normal time4(ms)',
                           'normal time5(ms)',
                           'normal time6(ms)', 'normal time7(ms)', 'normal time8(ms)', 'normal time9(ms)',
                           'normal time10(ms)'])


for shape1 in shape:
  for shape2 in shape:
    for _values_type in values_type:
      for _events_type in events_type:
        for _transpose in transpose:
          event_time_1, event_time_2, event_time_3, event_time_4, event_time_5, \
            event_time_6, event_time_7, event_time_8, event_time_9, event_time_10, \
            normal_time_1, normal_time_2, normal_time_3, normal_time_4, normal_time_5, \
            normal_time_6, normal_time_7, normal_time_8, normal_time_9, normal_time_10 = test_event_csrmv_dW(
            (shape1, shape2), _values_type, _events_type, _transpose)
          # append to dataframe
          df.loc[df.shape[0]] = [(shape1, shape2), 0.5, shape1, shape2, bm.get_platform(), _values_type, _events_type, _transpose,
                                 event_time_1, event_time_2, event_time_3, event_time_4,
                                 event_time_5,
                                 event_time_6, event_time_7, event_time_8, event_time_9,
                                 event_time_10,
                                 normal_time_1, normal_time_2, normal_time_3, normal_time_4, normal_time_5,
                                 normal_time_6, normal_time_7, normal_time_8, normal_time_9, normal_time_10]
          df.to_csv(f'{PATH}/csrmv_dW_{bm.get_platform()}.csv', index=False)
