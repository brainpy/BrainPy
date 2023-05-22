from time import time

import brainpy.math as bm
import jax.numpy as jnp
from jax import jit, vmap


def compare_jitconn_imp(platform='gpu'):
  bm.set_platform(platform)

  seed = 1234
  num_loop = 1

  all_shapes = [
    # (int(1e3), int(1e3)),
    # (int(1e3), int(1e4)),
    # (int(1e4), int(1e4)),
    # (int(5e4), int(5e4)),
    # (int(5e4), int(1e5)),
    # (int(5e5), int(1e5)),
    (int(5e5), int(5e5)),
    # (int(1e5), int(1e5)),
  ]

  for m in [32, 64, 128, 256]:
    for shape in all_shapes:
      for prob in [0.01]:
        print(f'm = {m}, shape = {shape}, prob = {prob}')
        f1 = jit(
          vmap(lambda a: bm.jitconn.mv_prob_normal(
            a, w_mu=0., w_sigma=0.01, conn_prob=prob, shape=shape, seed=seed, transpose=True
          ))
        )
        f2 = jit(lambda e: bm.jitconn.mm_prob_normal(
          e, w_mu=0., w_sigma=0.01, conn_prob=prob, shape=shape, seed=seed, version='v2'
        ))

        rng = bm.random.RandomState()
        mat = bm.as_jax(rng.random((m, shape[0])))
        r1 = f1(mat).block_until_ready()
        r2 = f2(mat).block_until_ready()
        assert r1.shape == r2.shape
        print(jnp.allclose(r1, r2))

        t0 = time()
        for _ in range(num_loop):
          f1(mat).block_until_ready()
        print(f'matvec vmap {time() - t0} s')

        t0 = time()
        for _ in range(num_loop):
          f2(mat).block_until_ready()
        print(f'matmat {time() - t0} s')

        print()
        bm.clear_buffer_memory()


if __name__ == '__main__':
  pass
  compare_jitconn_imp('gpu')
