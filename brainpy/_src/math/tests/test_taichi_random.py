# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.skip(reason="Skipped due to MacOS limitation, manual execution required for testing.")
import brainpy.math as bm
import taichi as ti
import matplotlib.pyplot as plt
import os
from brainpy._src.math.taichi_support import (taichi_lcg_rand as rand,
                                              taichi_uniform_int_distribution as randint,
                                              taichi_uniform_real_distribution as uniform,
                                              taichi_normal_distribution as normal,
                                              taichi_lfsr88,
                                              taichi_lfsr88_init, )

bm.set_platform('cpu')


def test_taichi_random():
  @ti.kernel
  def test_taichi_lfsr88(seed: ti.types.ndarray(ndim=1, dtype=ti.u32),
                         out: ti.types.ndarray(ndim=1, dtype=ti.f32)):
    seeds = taichi_lfsr88_init(seed[0])
    for i in range(out.shape[0]):
      seeds, result = taichi_lfsr88(seeds)
      out[i] = result

  @ti.kernel
  def test_taichi_lcg_rand(seed: ti.types.ndarray(ndim=1),
                           out: ti.types.ndarray(ndim=1)):
    for i in range(out.shape[0]):
      out[i] = rand(seed)

  @ti.kernel
  def test_taichi_uniform_int_distribution(seed: ti.types.ndarray(ndim=1),
                                           low_high: ti.types.ndarray(ndim=1),
                                           out: ti.types.ndarray(ndim=1)):
    seeds = taichi_lfsr88_init(seed[0])
    low = low_high[0]
    high = low_high[1]
    for i in range(out.shape[0]):
      seeds, result = taichi_lfsr88(seeds)
      out[i] = randint(result, low, high)

  @ti.kernel
  def test_taichi_uniform_real_distribution(seed: ti.types.ndarray(ndim=1),
                                            low_high: ti.types.ndarray(ndim=1),
                                            out: ti.types.ndarray(ndim=1)):
    seeds = taichi_lfsr88_init(seed[0])
    low = low_high[0]
    high = low_high[1]
    for i in range(out.shape[0]):
      seeds, result = taichi_lfsr88(seeds)
      out[i] = uniform(result, low, high)

  @ti.kernel
  def test_taichi_normal_distribution(seed: ti.types.ndarray(ndim=1),
                                      mu_sigma: ti.types.ndarray(ndim=1),
                                      out: ti.types.ndarray(ndim=1)):
    seeds = taichi_lfsr88_init(seed[0])
    mu = mu_sigma[0]
    sigma = mu_sigma[1]

    for i in range(out.shape[0]):
      seeds, result1 = taichi_lfsr88(seeds)
      seeds, result2 = taichi_lfsr88(seeds)
      out[i] = normal(result1, result2, mu, sigma)

  n = 100000
  seed = jnp.array([1234, ], dtype=jnp.uint32)
  low_high = jnp.array([0, 10])
  mu_sigma = jnp.array([0, 1])

  prim_lfsr88 = bm.XLACustomOp(cpu_kernel=test_taichi_lfsr88,
                               gpu_kernel=test_taichi_lfsr88)
  

  prim_lcg_rand = bm.XLACustomOp(cpu_kernel=test_taichi_lcg_rand,
                                 gpu_kernel=test_taichi_lcg_rand)
  prim_uniform_int_distribution = bm.XLACustomOp(cpu_kernel=test_taichi_uniform_int_distribution,
                                                 gpu_kernel=test_taichi_uniform_int_distribution)
  prim_uniform_real_distribution = bm.XLACustomOp(cpu_kernel=test_taichi_uniform_real_distribution,
                                                  gpu_kernel=test_taichi_uniform_real_distribution)
  prim_normal_distribution = bm.XLACustomOp(cpu_kernel=test_taichi_normal_distribution,
                                            gpu_kernel=test_taichi_normal_distribution)

  file_path = os.path.dirname(os.path.abspath(__file__))

  out = prim_lfsr88(seed, outs=[jax.ShapeDtypeStruct((n,), jnp.float32)])
  # show the distribution of out
  plt.hist(out, bins=100)
  plt.title("LFSR88 random number generator")
  plt.savefig(file_path + "/lfsr88.png")
  plt.close()

  out = prim_lcg_rand(seed,
                      outs=[jax.ShapeDtypeStruct((n,), jnp.float32)])
  # show the distribution of out
  plt.hist(out, bins=100)
  plt.title("LCG random number generator")
  plt.savefig(file_path + "/lcg_rand.png")
  plt.close()

  out = prim_uniform_int_distribution(seed, low_high,
                                      outs=[jax.ShapeDtypeStruct((n,), jnp.int32)])
  # show the distribution of out
  plt.hist(out, bins=10)
  plt.title("Uniform int distribution (0, 10)")
  plt.savefig(file_path + "/uniform_int_distribution.png")
  plt.close()

  out = prim_uniform_real_distribution(seed, low_high,
                                       outs=[jax.ShapeDtypeStruct((n,), jnp.float32)])
  # show the distribution of out
  plt.hist(out, bins=100)
  plt.title("Uniform real distribution (0, 10)")
  plt.savefig(file_path + "/uniform_real_distribution.png")
  plt.close()

  out = prim_normal_distribution(seed, mu_sigma,
                                 outs=[jax.ShapeDtypeStruct((n,), jnp.float32)])
  # show the distribution of out
  plt.title("Normal distribution mu=0, sigma=1")
  plt.hist(out, bins=100)
  plt.savefig(file_path + "/normal_distribution.png")
