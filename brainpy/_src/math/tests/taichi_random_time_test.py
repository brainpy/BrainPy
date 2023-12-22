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
                                             init_xorwow_seeds,
                                             taichi_lfsr88_0,
                                             taichi_xorwow_0,
                                             init_lfsr88_seeds_0,
                                             init_xorwow_seeds_0,)

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
        
@ti.kernel
def test_taichi_lfsr88_0(seed: ti.types.ndarray(ndim=1, dtype=ti.u32),
                        out: ti.types.ndarray(ndim=1, dtype=ti.f32)):
    s1, s2, s3, b = init_lfsr88_seeds_0(seed[0])
    for i in range(out.shape[0]):
        s1, s2, s3, b, result = taichi_lfsr88_0(s1, s2, s3, b)
        out[i] = result
    
@ti.kernel
def test_taichi_xorwow_0(seed: ti.types.ndarray(ndim=1, dtype=ti.u32),
                        out: ti.types.ndarray(ndim=1, dtype=ti.f32)):
    x, y, z, w, v, d = init_xorwow_seeds_0(seed[0])
    # print(seeds1, seeds2)
    for i in range(out.shape[0]):
        x, y, z, w, v, d, result = taichi_xorwow_0(x, y, z, w, v, d)
        out[i] = result

n = 100000000
seed = jnp.array([1234, ], dtype=jnp.uint32)

prim_lfsr88 = bm.XLACustomOp(cpu_kernel=test_taichi_lfsr88,
                               gpu_kernel=test_taichi_lfsr88)
  
prim_xorwow = bm.XLACustomOp(cpu_kernel=test_taichi_xorwow,
                            gpu_kernel=test_taichi_xorwow)

prim_lfsr88_0 = bm.XLACustomOp(cpu_kernel=test_taichi_lfsr88_0,
                               gpu_kernel=test_taichi_lfsr88_0)
  
prim_xorwow_0 = bm.XLACustomOp(cpu_kernel=test_taichi_xorwow_0,
                            gpu_kernel=test_taichi_xorwow_0)

out = jax.block_until_ready(prim_lfsr88(seed, outs=[jax.ShapeDtypeStruct((n,), jnp.float32)]))
time0 = time.time()
out = jax.block_until_ready(prim_lfsr88(seed, outs=[jax.ShapeDtypeStruct((n,), jnp.float32)]))
time1 = time.time()

out = jax.block_until_ready(prim_xorwow(seed, outs=[jax.ShapeDtypeStruct((n,), jnp.float32)]))
time2 = time.time()
out = jax.block_until_ready(prim_xorwow(seed, outs=[jax.ShapeDtypeStruct((n,), jnp.float32)]))
time3 = time.time()

out = jax.block_until_ready(prim_lfsr88_0(seed, outs=[jax.ShapeDtypeStruct((n,), jnp.float32)]))
time4 = time.time()
out = jax.block_until_ready(prim_lfsr88_0(seed, outs=[jax.ShapeDtypeStruct((n,), jnp.float32)]))
time5 = time.time()

out = jax.block_until_ready(prim_xorwow_0(seed, outs=[jax.ShapeDtypeStruct((n,), jnp.float32)]))
time6 = time.time()
out = jax.block_until_ready(prim_xorwow_0(seed, outs=[jax.ShapeDtypeStruct((n,), jnp.float32)]))
time7 = time.time()

print('lfsr88: ', time1 - time0)
print('xorwow: ', time3 - time2)
print('lfsr88_0: ', time5 - time4)
print('xorwow_0: ', time7 - time6)
