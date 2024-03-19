import jax
import jax.numpy as jnp
import pytest

import brainpy.math as bm
from brainpy._src.dependency_check import import_cupy, import_cupy_jit

cp = import_cupy(error_if_not_found=False)
cp_jit = import_cupy_jit(error_if_not_found=False)
if cp is None:
  pytest.skip('no cupy', allow_module_level=True)
bm.set_platform('gpu')


def test_cupy_based():
  # Raw Module

  source_code = r'''
  extern "C"{
  
  __global__ void kernel(const float* x1, const float* x2, unsigned int N, float* y)
  {
      unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
      if (tid < N)
      {
          y[tid] = x1[tid] + x2[tid];
      }
  }
  }
  '''
  N = 10
  x1 = bm.ones((N, N))
  x2 = bm.ones((N, N))
  prim1 = bm.XLACustomOp(gpu_kernel=source_code)

  # n = jnp.asarray([N**2,], dtype=jnp.int32)

  y = prim1(x1, x2, N ** 2, grid=(N,), block=(N,), outs=[jax.ShapeDtypeStruct((N, N), dtype=jnp.float32)])[0]

  print(y)
  assert jnp.allclose(y, x1 + x2)

  # JIT Kernel
  @cp_jit.rawkernel()
  def elementwise_copy(x, size, y):
    tid = cp_jit.blockIdx.x * cp_jit.blockDim.x + cp_jit.threadIdx.x
    ntid = cp_jit.gridDim.x * cp_jit.blockDim.x
    for i in range(tid, size, ntid):
      y[i] = x[i]

  size = 100
  x = bm.ones((size,))

  prim2 = bm.XLACustomOp(gpu_kernel=elementwise_copy)

  y = prim2(x, size, grid=(10,), block=(10,), outs=[jax.ShapeDtypeStruct((size,), dtype=jnp.float32)])[0]

  print(y)
  assert jnp.allclose(y, x)

# test_cupy_based()
