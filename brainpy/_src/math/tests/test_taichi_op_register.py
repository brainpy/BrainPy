import unittest
import jax
import jax.numpy as jnp
import taichi as ti
import os
taichi_path = ti.__path__[0]
taichi_c_api_install_dir = os.path.join(taichi_path, '_lib', 'c_api')
taichi_lib_dir = os.path.join(taichi_path, '_lib', 'runtime')
os.environ.update({
'TAICHI_C_API_INSTALL_DIR': taichi_c_api_install_dir,
'TI_LIB_DIR': taichi_lib_dir
})

import brainpy.math as bm

# from brainpylib import cpu_ops
# print(cpu_ops.registrations().items())

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
