# -*- coding: utf-8 -*-

import timeit
import time
import jax
import jax.numpy as jnp
import numpy as np
from brainpylib.event_sum import _event_sum_prim

import brainpy as bp
import brainpy.math as bm

# from brainpy.building import

bm.set_platform('gpu')

# def test1():
bm.random.seed(123)
size = 30000
conn = bp.conn.FixedProb(prob=0.5, seed=123)
conn(pre_size=size, post_size=size)
# pre2post = conn.require('pre2post')
pre_ids, post_ids = conn.require('pre_ids', 'post_ids')
print("pre_ids size:", pre_ids.size)
# indices = jnp.arange(size, dtype=jnp.uint32)
# idnptr = jnp.arange(size + 1, dtype=jnp.uint32)
sps = bm.random.randint(0, 2, size).value < 1
value = 2.


# f = jax.jit(event_sum)


# @partial(jax.jit, static_argnums=2)
# def f(sps, pre2post, size, value):
#   return event_sum(sps, pre2post, size, value)


@jax.jit
def ours(events):
  out = jnp.zeros(size)
  out = _event_sum_prim.bind(events, post_ids.value, pre_ids.value, jnp.zeros(1), out)
  # print(type(out), out)
  # print(type(value), value)
  return out


@jax.jit
def yours(events):
  out = jnp.zeros(size)
  out = out.at[post_ids.value].add(events[pre_ids.value])
  return out


a = ours(sps)
b = np.asarray(a)
# print(b)
print(b.size)

# a = yours(sps)
# b = np.asarray(a)
# # print(b)
# print(b.size)

sps = bm.random.randint(0, 2, size).value < 1

t0 = time.time()
ours(sps)
print(time.time() - t0)


# t0 = time.time()
# yours(sps)
# print(time.time() - t0)

# %timeit f(sps, (indices, indptr), size, value)
# print(timeit.timeit('ours(sps, value)', globals=globals()))
# print(timeit.timeit('yours(sps, value)', globals=globals()))
