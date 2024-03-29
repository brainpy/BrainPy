import jax
import jax.numpy as jnp
import pytest

import brainpy.math as bm
from brainpy._src.dependency_check import import_taichi

ti = import_taichi(error_if_not_found=False)
if ti is None:
  pytest.skip('no taichi', allow_module_level=True)

bm.set_platform('cpu')

@ti.func
def get_weight(weight: ti.types.ndarray(ndim=0)) -> ti.f32:
  return weight[None]


@ti.func
def update_output(out: ti.types.ndarray(ndim=1), index: ti.i32, weight_val: ti.f32):
  out[index] += weight_val


@ti.kernel
def event_ell_cpu(indices: ti.types.ndarray(ndim=2),
                  vector: ti.types.ndarray(ndim=1),
                  weight: ti.types.ndarray(ndim=0),
                  out: ti.types.ndarray(ndim=1)):
  weight_val = get_weight(weight)
  num_rows, num_cols = indices.shape
  ti.loop_config(serialize=True)
  for i in range(num_rows):
    if vector[i]:
      for j in range(num_cols):
        update_output(out, indices[i, j], weight_val)

@ti.kernel
def event_ell_gpu(indices: ti.types.ndarray(ndim=2),
                  vector: ti.types.ndarray(ndim=1),
                  weight: ti.types.ndarray(ndim=0),
                  out: ti.types.ndarray(ndim=1)):
  weight_val = get_weight(weight)
  num_rows, num_cols = indices.shape
  for i in range(num_rows):
    if vector[i]:
      for j in range(num_cols):
        update_output(out, indices[i, j], weight_val)

prim = bm.XLACustomOp(cpu_kernel=event_ell_cpu, gpu_kernel=event_ell_gpu)


def test_taichi_op_register():
  s = 1000
  indices = bm.random.randint(0, s, (s, 1000))
  vector = bm.random.rand(s) < 0.1

  out = prim(indices, vector, 1.0, outs=[jax.ShapeDtypeStruct((s,), dtype=jnp.float32)])

  out = prim(indices, vector, 1.0, outs=[jax.ShapeDtypeStruct((s,), dtype=jnp.float32)])

  print(out)

# test_taichi_op_register()
