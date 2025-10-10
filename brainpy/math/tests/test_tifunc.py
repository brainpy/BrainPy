# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.skip(reason="Skipped due to MacOS limitation, manual execution required for testing.")
import brainpy.math as bm
import matplotlib.pyplot as plt
import os

bm.set_platform('cpu')


def test_taichi_random():
    @ti.kernel
    def test_taichi_lfsr88(seed: ti.types.ndarray(ndim=1, dtype=ti.u32),
                           out: ti.types.ndarray(ndim=1, dtype=ti.f32)):
        key = bm.tifunc.lfsr88_key(seed[0])
        for i in range(out.shape[0]):
            key, result = bm.tifunc.lfsr88_rand(key)
            out[i] = result

    @ti.kernel
    def test_taichi_lcg_rand(seed: ti.types.ndarray(ndim=1),
                             out: ti.types.ndarray(ndim=1)):
        for i in range(out.shape[0]):
            out[i] = bm.tifunc.taichi_lcg_rand(seed)

    @ti.kernel
    def test_taichi_uniform_int_distribution(seed: ti.types.ndarray(ndim=1),
                                             low_high: ti.types.ndarray(ndim=1),
                                             out: ti.types.ndarray(ndim=1)):
        key = bm.tifunc.lfsr88_key(seed[0])
        low = low_high[0]
        high = low_high[1]
        for i in range(out.shape[0]):
            key, out[i] = bm.tifunc.lfsr88_randint(key, low, high)

    @ti.kernel
    def test_taichi_uniform_real_distribution(seed: ti.types.ndarray(ndim=1),
                                              low_high: ti.types.ndarray(ndim=1),
                                              out: ti.types.ndarray(ndim=1)):
        key = bm.tifunc.lfsr88_key(seed[0])
        low = low_high[0]
        high = low_high[1]
        for i in range(out.shape[0]):
            key, out[i] = bm.tifunc.lfsr88_uniform(key, low, high)

    @ti.kernel
    def test_taichi_normal_distribution(seed: ti.types.ndarray(ndim=1),
                                        mu_sigma: ti.types.ndarray(ndim=1),
                                        out: ti.types.ndarray(ndim=1)):
        key = bm.tifunc.lfsr88_key(seed[0])
        mu = mu_sigma[0]
        sigma = mu_sigma[1]

        for i in range(out.shape[0]):
            key, out[i] = bm.tifunc.lfsr88_normal(key, mu, sigma)

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

# TODO; test default types
