from time import time

import brainpy.math as bm
from jax import jit


def compare_jitconn_imp(platform='gpu'):
  bm.set_platform(platform)

  weight = 1.
  seed = 1234

  all_shapes = [
    # (int(1e3), int(1e3)),
    # (int(1e3), int(1e4)),
    # (int(1e4), int(1e4)),
    # (int(5e4), int(5e4)),
    # (int(5e4), int(1e5)),
    (int(5e5), int(1e5)),
    (int(5e5), int(5e5)),
  ]

  for shape in all_shapes:
    for prob in [0.01, 0.05, 0.1, 0.2, 0.4, 0.8]:
      for transpose in [True, False]:
        print(f'shape = {shape}, prob = {prob}, transpose = {transpose}')
        f1 = jit(lambda e: bm.jitconn.mv_prob_homo(
          e, weight, conn_prob=prob, shape=shape, seed=seed, transpose=transpose))
        f2 = jit(lambda e: bm.jitconn.mv_prob_homo(
          e, weight, conn_prob=prob, shape=shape, seed=seed, transpose=transpose))

        rng = bm.random.RandomState()
        events = bm.as_jax(rng.random(shape[0] if transpose else shape[1]))
        f1(events).block_until_ready()
        f2(events).block_until_ready()

        t0 = time()
        for _ in range(100):
          f1(events).block_until_ready()
        print(f'event_matvec_v1 {time() - t0} s')

        t0 = time()
        for _ in range(100):
          f2(events).block_until_ready()
        print(f'event_matvec_v2 {time() - t0} s')

        print()
        bm.clear_buffer_memory()


if __name__ == '__main__':
  pass
  compare_jitconn_imp('gpu')
