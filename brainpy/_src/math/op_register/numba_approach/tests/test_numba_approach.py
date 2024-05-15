import jax.core
import pytest
from jax.core import ShapedArray

import brainpy.math as bm
from brainpy._src.dependency_check import import_numba

numba = import_numba(error_if_not_found=False)
if numba is None:
  pytest.skip('no numba', allow_module_level=True)

bm.set_platform('cpu')


def eval_shape(a):
  b = ShapedArray(a.shape, dtype=a.dtype)
  return b

@numba.njit(parallel=True)
def con_compute(outs, ins):
  b = outs
  a = ins
  b[:] = a + 1

def test_CustomOpByNumba_single_result():
  op = bm.CustomOpByNumba(eval_shape, con_compute, multiple_results=False)
  print(op(bm.zeros(10)))

def eval_shape2(a, b):
  c = ShapedArray(a.shape, dtype=a.dtype)
  d = ShapedArray(b.shape, dtype=b.dtype)
  return c, d


def con_compute2(outs, ins):
  # c = outs[0]  # take out all the outputs
  # d = outs[1]
  # a = ins[0]  # take out all the inputs
  # b = ins[1]
  c, d = outs
  a, b = ins
  c[:] = a + 1
  d[:] = b * 2

def test_CustomOpByNumba_multiple_results():
  op2 = bm.CustomOpByNumba(eval_shape2, con_compute2, multiple_results=True)
  print(op2(bm.zeros(10), bm.ones(10)))

test_CustomOpByNumba_multiple_results()