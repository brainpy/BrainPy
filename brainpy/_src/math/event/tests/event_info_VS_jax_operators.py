from time import time

import brainpy.math as bm
from jax import jit, vmap, numpy as jnp

from brainpylib import event_info


def compare_argsort_and_sum(platform='cpu'):
  """

  CPU
  ---

  shape = (100, 10000)
  brainpylib        0.1872694492340088 s
  JAX argsort + sum 5.297466516494751 s

  shape = (100, 100000)
  brainpylib        2.333505153656006 s
  JAX argsort + sum 65.20281910896301 s

  shape = (1000, 10000)
  brainpylib        2.0739688873291016 s
  JAX argsort + sum 53.70602822303772 s

  shape = (10000, 1000)
  brainpylib        1.7262670993804932 s
  JAX argsort + sum 43.92174816131592 s

  GPU
  ---
  shape = (100, 100000)
  brainpylib        0.14670848846435547 s
  JAX argsort + sum 1.001936435699463 s

  shape = (100, 1000000)
  brainpylib        0.27660632133483887 s
  JAX argsort + sum 16.390073776245117 s

  shape = (1000, 100000)
  brainpylib        0.2619345188140869 s
  JAX argsort + sum 9.715844869613647 s

  shape = (1000, 500000)
  brainpylib        1.201209306716919 s
  JAX argsort + sum 71.19761657714844 s

  """

  bm.set_platform(platform)

  rng = bm.random.RandomState(123)
  bp_event_info = jit(vmap(event_info))
  jax_event_info = jit(vmap(lambda events: (jnp.argsort(events), jnp.sum(events))))

  if platform == 'cpu':
    all_shapes = [
      (100, 10000),
      (100, 100000),
      (1000, 10000),
      (10000, 1000),
    ]
  else:
    all_shapes = [
      (100, 100000),
      (100, 1000000),
      (1000, 100000),
      (1000, 500000),
    ]

  for shape in all_shapes:
    print(f'shape = {shape}')

    events = rng.random(shape).value < 0.1
    event_ids1, event_num1 = bp_event_info(events)
    event_ids2, event_num2 = jax_event_info(events)
    assert jnp.allclose(event_num1, event_num2)
    event_ids1.block_until_ready()
    event_ids2.block_until_ready()

    t0 = time()
    for _ in range(100):
      a, b = bp_event_info(events)
      r = a.block_until_ready()
    print(f'brainpylib        {time() - t0} s')

    t0 = time()
    for _ in range(100):
      a, b = jax_event_info(events)
      r = a.block_until_ready()
    print(f'JAX argsort + sum {time() - t0} s')

    print()


def compare_argsort(platform='cpu'):
  """

  CPU
  ---

  shape = (100, 10000)
  brainpylib  0.19738531112670898 s
  JAX argsort 5.301469087600708 s

  shape = (100, 100000)
  brainpylib  2.3321938514709473 s
  JAX argsort 65.13460850715637 s

  shape = (1000, 10000)
  brainpylib  2.0956876277923584 s
  JAX argsort 53.863110065460205 s

  shape = (10000, 1000)
  brainpylib  1.7127799987792969 s
  JAX argsort 44.05547475814819 s

  GPU
  ---
  shape = (100, 100000)
  brainpylib  0.1415419578552246 s
  JAX argsort 0.9982438087463379 s

  shape = (100, 1000000)
  brainpylib  0.3224947452545166 s
  JAX argsort 16.504750967025757 s

  shape = (1000, 100000)
  brainpylib  0.2781648635864258 s
  JAX argsort 9.691488981246948 s

  shape = (1000, 500000)
  brainpylib  1.2167487144470215 s
  JAX argsort 71.68716263771057 s

  """

  bm.set_platform(platform)

  rng = bm.random.RandomState(123)
  bp_event_info = jit(vmap(event_info))
  jax_event_info = jit(vmap(lambda events: jnp.argsort(events)))

  if platform == 'cpu':
    all_shapes = [
      (100, 10000),
      (100, 100000),
      (1000, 10000),
      (10000, 1000),
    ]
  else:
    all_shapes = [
      (100, 100000),
      (100, 1000000),
      (1000, 100000),
      (1000, 500000),
    ]

  for shape in all_shapes:
    print(f'shape = {shape}')

    events = rng.random(shape).value < 0.1
    event_ids1, event_num1 = bp_event_info(events)
    event_ids1.block_until_ready()
    event_ids2 = jax_event_info(events)
    event_ids2.block_until_ready()

    t0 = time()
    for _ in range(100):
      a, b = bp_event_info(events)
      r = a.block_until_ready()
    print(f'brainpylib  {time() - t0} s')

    t0 = time()
    for _ in range(100):
      a = jax_event_info(events)
      r = a.block_until_ready()
    print(f'JAX argsort {time() - t0} s')

    print()


def compare_where(platform='cpu'):
  """

  CPU
  ---

  shape = (100, 10000)
  brainpylib 0.20480966567993164 s
  JAX where  0.7068588733673096 s

  shape = (100, 100000)
  brainpylib 2.3373026847839355 s
  JAX where  5.862265348434448 s

  shape = (1000, 10000)
  brainpylib 2.105764865875244 s
  JAX where  5.914586067199707 s

  shape = (10000, 1000)
  brainpylib 1.724682331085205 s
  JAX where  5.718563795089722 s

  GPU
  ---
  shape = (100, 100000)
  brainpylib 0.15492558479309082 s
  JAX where  0.3146538734436035 s

  shape = (100, 1000000)
  brainpylib 0.3290700912475586 s
  JAX where  1.7064015865325928 s

  shape = (1000, 100000)
  brainpylib 0.2895216941833496 s
  JAX where  1.6910102367401123 s

  shape = (1000, 500000)
  brainpylib 1.173649787902832 s
  JAX where  7.868000268936157 s

  """

  bm.set_platform(platform)

  rng = bm.random.RandomState(123)
  bp_event_info = jit(vmap(event_info))
  jax_event_info = jit(vmap(lambda events: jnp.where(events, size=events.shape[0])))

  if platform == 'cpu':
    all_shapes = [
      (100, 10000),
      (100, 100000),
      (1000, 10000),
      (10000, 1000),
    ]
  else:
    all_shapes = [
      (100, 100000),
      (100, 1000000),
      (1000, 100000),
      (1000, 500000),
    ]

  for shape in all_shapes:
    print(f'shape = {shape}')

    events = rng.random(shape).value < 0.1
    event_ids1, event_num1 = bp_event_info(events)
    event_ids1.block_until_ready()
    event_ids2, = jax_event_info(events)
    event_ids2.block_until_ready()

    t0 = time()
    for _ in range(100):
      a, b = bp_event_info(events)
      r = a.block_until_ready()
    print(f'brainpylib {time() - t0} s')

    t0 = time()
    for _ in range(100):
      a,  = jax_event_info(events)
      r = a.block_until_ready()
    print(f'JAX where  {time() - t0} s')

    print()


if __name__ == '__main__':
  # compare_argsort_and_sum('cpu')
  # compare_argsort_and_sum('gpu')
  # compare_argsort('cpu')
  compare_argsort('gpu')
  # compare_where('cpu')
  # compare_where('gpu')
