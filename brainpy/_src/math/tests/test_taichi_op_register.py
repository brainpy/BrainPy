import unittest
import jax
import jax.numpy as jnp
import brainpy.math as bm
import taichi as ti
import os

bm.set_platform('gpu')

@ti.kernel
def event_ell_cpu(indices: ti.types.ndarray(ndim=2), vector: ti.types.ndarray(ndim=1), weight: ti.types.ndarray(ndim=1), out: ti.types.ndarray(ndim=1)):
    weight_0 = weight[0]
    num_rows, num_cols = indices.shape
    ti.loop_config(serialize=True)
    for i in range(num_rows):
        if vector[i]:
            for j in range(num_cols):
                out[indices[i, j]] += weight_0

prim = bm.XLACustomOp(gpu_kernel=event_ell_cpu)

def test_taichi_op_register():
    s = 1000
    indices = bm.random.randint(0, s, (s, 1000))
    vector = bm.random.rand(s) < 0.1
    weight = bm.array([1.0])

    out = prim(indices, vector, weight, outs=[jax.ShapeDtypeStruct((s, ), dtype=jnp.float32)])

    print(out)

test_taichi_op_register()