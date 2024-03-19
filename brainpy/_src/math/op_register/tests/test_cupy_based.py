import jax
import pytest

import brainpy.math as bm
from brainpy._src.dependency_check import import_cupy, import_cupy_jit, import_taichi

cp = import_cupy(error_if_not_found=False)
cp_jit = import_cupy_jit(error_if_not_found=False)
ti = import_taichi(error_if_not_found=False)
if cp is None or ti is None:
  pytest.skip('no cupy or taichi', allow_module_level=True)
bm.set_platform('cpu')


def test_cupy_based():
  bm.op_register.clear_taichi_aot_caches()
  # Raw Module

  @ti.kernel
  def simpleAdd(x1: ti.types.ndarray(ndim=2),
                x2: ti.types.ndarray(ndim=2),
                n: ti.types.ndarray(ndim=0),
                y: ti.types.ndarray(ndim=2)):
    for i, j in y:
      y[i, j] = x1[i, j] + x2[i, j]

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

  mod = cp.RawModule(code=source_code)
  kernel = mod.get_function('kernel')

  prim1 = bm.XLACustomOp(cpu_kernel=simpleAdd, gpu_kernel=kernel)

  y = prim1(x1, x2, N**2, grid=(N,), block=(N,), outs=[jax.ShapeDtypeStruct((N, N), dtype=bm.float32)])[0]

  print(y)
  assert bm.allclose(y, x1 + x2)

  # JIT Kernel
  @ti.kernel
  def elementwise_copy_taichi(x: ti.types.ndarray(ndim=1),
                              size: ti.types.ndarray(ndim=1),
                              y: ti.types.ndarray(ndim=1)):
    for i in y:
      y[i] = x[i]

  @cp_jit.rawkernel()
  def elementwise_copy(x, size, y):
    tid = cp_jit.blockIdx.x * cp_jit.blockDim.x + cp_jit.threadIdx.x
    ntid = cp_jit.gridDim.x * cp_jit.blockDim.x
    for i in range(tid, size, ntid):
      y[i] = x[i]

  size = 100
  x = bm.ones((size,))

  prim2 = bm.XLACustomOp(cpu_kernel=elementwise_copy_taichi, gpu_kernel=elementwise_copy)

  y = prim2(x, size, grid=(10,), block=(10,), outs=[jax.ShapeDtypeStruct((size,), dtype=bm.float32)])[0]

  print(y)
  assert bm.allclose(y, x)

# test_cupy_based()
