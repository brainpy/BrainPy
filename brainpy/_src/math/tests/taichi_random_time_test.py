import jax
import jax.numpy as jnp
import time

import brainpy.math as bm
import taichi as ti
import matplotlib.pyplot as plt

from brainpy._src.math.taichi_random import (taichi_lcg_rand as rand,
                                             taichi_uniform_int_distribution as randint,
                                             taichi_uniform_real_distribution as uniform,
                                             taichi_normal_distribution as normal,
                                             taichi_lfsr88,
                                             init_lfsr88_seeds,
                                             taichi_xorwow,
                                             init_xorwow_seeds)

bm.set_platform('gpu')

@ti.kernel
def test_taichi_lfsr88(seed: ti.types.ndarray(ndim=1, dtype=ti.u32),
                        out: ti.types.ndarray(ndim=1, dtype=ti.f32)):
    seeds = init_lfsr88_seeds(seed[0])
    for i in range(out.shape[0]):
        seeds, result = taichi_lfsr88(seeds)
        out[i] = result
    
@ti.kernel
def test_taichi_xorwow(seed: ti.types.ndarray(ndim=1, dtype=ti.u32),
                        out: ti.types.ndarray(ndim=1, dtype=ti.f32)):
    seeds1, seeds2 = init_xorwow_seeds(seed[0])
    # print(seeds1, seeds2)
    for i in range(out.shape[0]):
        seeds1, seeds2, result = taichi_xorwow(seeds1, seeds2)
        out[i] = result
        
n = 100000000
seed = jnp.array([1234, ], dtype=jnp.uint32)

prim_lfsr88 = bm.XLACustomOp(cpu_kernel=test_taichi_lfsr88,
                               gpu_kernel=test_taichi_lfsr88)
  
prim_xorwow = bm.XLACustomOp(cpu_kernel=test_taichi_xorwow,
                            gpu_kernel=test_taichi_xorwow)

out = jax.block_until_ready(prim_lfsr88(seed, outs=[jax.ShapeDtypeStruct((n,), jnp.float32)]))
time0 = time.time()
out = jax.block_until_ready(prim_lfsr88(seed, outs=[jax.ShapeDtypeStruct((n,), jnp.float32)]))
time1 = time.time()

out = jax.block_until_ready(prim_xorwow(seed, outs=[jax.ShapeDtypeStruct((n,), jnp.float32)]))
time2 = time.time()
out = jax.block_until_ready(prim_xorwow(seed, outs=[jax.ShapeDtypeStruct((n,), jnp.float32)]))
time3 = time.time()


print('lfsr88: ', time1 - time0)
print('xorwow: ', time3 - time2)
