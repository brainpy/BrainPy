# -*- coding: utf-8 -*-

from functools import partial

import jax
import jax.numpy as jnp
import pytest
pytestmark = pytest.mark.skip(reason="Skipped due to MacOS limitation, manual execution required for testing.")
import platform
import brainpy as bp
import brainpy.math as bm
import taichi as ti
import matplotlib.pyplot as plt
import os
from brainpy.math.random import (taichi_lcg_rand as rand,
                                 taichi_uniform_int_distribution as randint,
                                 taichi_uniform_real_distribution as uniform,
                                 taichi_normal_distribution as normal,)

bm.set_platform('cpu')

@ti.kernel
def test_taichi_lcg_rand(seed: ti.types.ndarray(ndim=1),
                         out: ti.types.ndarray(ndim=1)):
    rand(seed, out)

@ti.kernel
def test_taichi_uniform_int_distribution(seed: ti.types.ndarray(ndim=1),
                                         low_high: ti.types.ndarray(ndim=1),
                                         random_sequence: ti.types.ndarray(ndim=1),
                                         out: ti.types.ndarray(ndim=1)):
    low = low_high[0]
    high = low_high[1]
    n = out.shape[0]
    rand(seed, random_sequence)
    for i in range(n):
        out[i] = randint(random_sequence[i], low, high)

@ti.kernel
def test_taichi_uniform_real_distribution(seed: ti.types.ndarray(ndim=1),
                                          low_high: ti.types.ndarray(ndim=1),
                                          random_sequence: ti.types.ndarray(ndim=1),
                                          out: ti.types.ndarray(ndim=1)):
    low = low_high[0]
    high = low_high[1]
    n = out.shape[0]
    rand(seed, random_sequence)
    for i in range(n):
        out[i] = uniform(random_sequence[i], low, high)

@ti.kernel
def test_taichi_normal_distribution(seed: ti.types.ndarray(ndim=1),
                                    mu_sigma: ti.types.ndarray(ndim=1),
                                    random_sequence: ti.types.ndarray(ndim=1),
                                    out: ti.types.ndarray(ndim=1)):
    mu = mu_sigma[0]
    sigma = mu_sigma[1]
    n = out.shape[0]
    rand(seed, random_sequence)
    for i in range(n):
        out[i] = normal(random_sequence[2 * i], random_sequence[2 * i + 1] , mu, sigma)


n = 100000
seed = jnp.array([1234,], dtype=jnp.uint32)
low_high = jnp.array([0, 10])
mu_sigma = jnp.array([0, 1])

prim_lcg_rand = bm.XLACustomOp(cpu_kernel=test_taichi_lcg_rand, 
                               gpu_kernel=test_taichi_lcg_rand)
prim_uniform_int_distribution = bm.XLACustomOp(cpu_kernel=test_taichi_uniform_int_distribution, 
                                               gpu_kernel=test_taichi_uniform_int_distribution)
prim_uniform_real_distribution = bm.XLACustomOp(cpu_kernel=test_taichi_uniform_real_distribution, 
                                                gpu_kernel=test_taichi_uniform_real_distribution)
prim_normal_distribution = bm.XLACustomOp(cpu_kernel=test_taichi_normal_distribution, 
                                          gpu_kernel=test_taichi_normal_distribution)

file_path = os.path.dirname(os.path.abspath(__file__))

out = prim_lcg_rand(seed,
                    outs=[jax.ShapeDtypeStruct((n,), jnp.float32)])
# show the distribution of out
plt.hist(out, bins=100)
plt.savefig(file_path + "/lcg_rand.png")
plt.close()

out = prim_uniform_int_distribution(seed, low_high, jnp.zeros((n,), dtype=jnp.float32),
                                    outs=[jax.ShapeDtypeStruct((n,), jnp.int32)])
# show the distribution of out
plt.hist(out, bins=10)
plt.savefig(file_path + "/uniform_int_distribution.png")
plt.close()

out = prim_uniform_real_distribution(seed, low_high, jnp.zeros((n,), dtype=jnp.float32),
                                        outs=[jax.ShapeDtypeStruct((n,), jnp.float32)])
# show the distribution of out
plt.hist(out, bins=100)
plt.savefig(file_path + "/uniform_real_distribution.png")
plt.close()

out = prim_normal_distribution(seed, mu_sigma, jnp.zeros((2 * n,), dtype=jnp.float32),
                                 outs=[jax.ShapeDtypeStruct((n,), jnp.float32)])
# show the distribution of out
plt.hist(out, bins=100)
plt.savefig(file_path + "/normal_distribution.png")





