import jax
import jax.numpy as jnp

import brainpy.math as bm
import taichi as ti

from brainpy._src.dependency_check import import_taichi
ti = import_taichi(error_if_not_found=False)
if ti is None:
  import pytest
  pytest.skip('no taichi', allow_module_level=True)


@ti.func
def get_weight(weight: ti.types.ndarray(ndim=1)) -> ti.f32:
  return weight[0]


@ti.func
def update_output(out: ti.types.ndarray(ndim=1), index: ti.i32, weight_val: ti.f32):
  out[index] += weight_val


@ti.kernel
def event_ell_cpu(indices: ti.types.ndarray(ndim=2),
                  vector: ti.types.ndarray(ndim=1),
                  weight: ti.types.ndarray(ndim=1),
                  out: ti.types.ndarray(ndim=1)):
  weight_val = get_weight(weight)
  num_rows, num_cols = indices.shape
  ti.loop_config(serialize=True)
  for i in range(num_rows):
    if vector[i]:
      for j in range(num_cols):
        update_output(out, indices[i, j], weight_val)


prim = bm.XLACustomOp(cpu_kernel=event_ell_cpu)


def test_taichi_clean_cache():
  s = 1000
  indices = bm.random.randint(0, s, (s, 100))
  vector = bm.random.rand(s) < 0.1
  weight = bm.array([1.0])

  out = prim(indices, vector, weight, outs=[jax.ShapeDtypeStruct((s,), dtype=jnp.float32)])

  out = prim(indices, vector, weight, outs=[jax.ShapeDtypeStruct((s,), dtype=jnp.float32)])

  print(out)
  bm.clear_buffer_memory()

  print('kernels: ', bm.count_taichi_aot_kernels())

  bm.clear_taichi_aot_caches()

  print('kernels: ', bm.count_taichi_aot_kernels())

# test_taichi_clean_cache()
