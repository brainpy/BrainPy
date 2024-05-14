import jax.core
import pytest
from jax.core import ShapedArray

import brainpy.math as bm
from brainpy._src.dependency_check import import_numba

numba = import_numba(error_if_not_found=False)
if numba is None:
  pytest.skip('no numba', allow_module_level=True)

bm.set_platform('cpu')


@numba.njit(fastmath=True)
def numba_event_csrmv(weight, indices, vector, outs):
  outs.fill(0)
  weight = weight[()]  # 0d
  for row_i in range(vector.shape[0]):
    if vector[row_i]:
      for j in indices[row_i]:
        outs[j] += weight


prim = bm.XLACustomOp(numba_event_csrmv)


def call(s=100):
  indices = bm.random.randint(0, s, (s, 80))
  vector = bm.random.rand(s) < 0.1
  out = prim(1., indices, vector, outs=[jax.ShapeDtypeStruct([s], dtype=bm.float32)])
  print(out[0].shape)


def test_event_ELL():
  call(1000)
  call(100)
  bm.clear_buffer_memory()

# CustomOpByNumba Test

def eval_shape(a):
  b = ShapedArray(a.shape, dtype=a.dtype)
  return b

@numba.njit(parallel=True)
def con_compute(outs, ins):
  b = outs
  a = ins
  b[:] = a + 1

def test_CustomOpByNumba():
  op = bm.CustomOpByNumba(eval_shape, con_compute, multiple_results=False)
  print(op(bm.zeros(10)))
  assert bm.allclose(op(bm.zeros(10)), bm.ones(10))