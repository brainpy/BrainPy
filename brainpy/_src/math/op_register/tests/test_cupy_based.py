import jax.numpy as jnp
import jax
import cupy as cp
from time import time

import brainpy.math as bm
from brainpy._src.math import as_jax

bm.set_platform('gpu')


def test_cupy_based():
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
  x1 = bm.random.rand(N, N)
  # x1_cp = cp.from_dlpack(jax.dlpack.to_dlpack(as_jax(x1)))
  x2 = bm.ones((N, N))
  # x2_cp = cp.from_dlpack(jax.dlpack.to_dlpack(as_jax(x2)))
  y = bm.zeros((N, N))
  # y_cp = cp.from_dlpack(jax.dlpack.to_dlpack(as_jax(y)))

  # mod = cp.RawModule(code=source_code)
  # kernel = mod.get_function('kernel')
  # y = kernel((N,), (N,), (x1_cp, x2_cp, N**2, y_cp))
  # print(y_cp)

  prim = bm.XLACustomOp(gpu_kernel=source_code)

  n = jnp.asarray([N**2,], dtype=jnp.int32)

  y = prim(x1, x2, n, grid=(N,), block=(N,), outs=[jax.ShapeDtypeStruct((N, N), dtype=jnp.float32)])

  print(y)
  assert jnp.allclose(y, x1 + x2)

  # N = 10
  # x1 = cp.arange(N**2, dtype=cp.float32).reshape(N, N)
  # x2 = cp.ones((N, N), dtype=cp.float32)
  # y = cp.zeros((N, N), dtype=cp.float32)
  # ker_sum((N,), (N,), (x1, x2, y, N**2))   # y = x1 + x2
  # assert cp.allclose(y, x1 + x2)
  # ker_times((N,), (N,), (x1, x2, y, N**2)) # y = x1 * x2
  # assert cp.allclose(y, x1 * x2)


test_cupy_based()
