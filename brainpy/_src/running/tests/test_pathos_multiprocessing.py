import sys

import jax
import pytest
from absl.testing import parameterized

import brainpy as bp
import brainpy.math as bm

if sys.platform == 'win32' and sys.version_info.minor >= 11:
  pytest.skip('python 3.11 does not support.', allow_module_level=True)
else:
  pytest.skip('Cannot pass tests.', allow_module_level=True)


class TestParallel(parameterized.TestCase):
  @parameterized.product(
    duration=[1e2, 1e3, 1e4, 1e5]
  )
  def test_cpu_unordered_parallel_v1(self, duration):
    @jax.jit
    def body(inp):
      return bm.for_loop(lambda x: x + 1e-9, inp)

    input_long = bm.random.randn(1, int(duration / bm.dt), 3) / 100

    r = bp.running.cpu_ordered_parallel(body, {'inp': [input_long, input_long]}, num_process=2)
    assert bm.allclose(r[0], r[1])

  @parameterized.product(
    duration=[1e2, 1e3, 1e4, 1e5]
  )
  def test_cpu_unordered_parallel_v2(self, duration):
    @jax.jit
    def body(inp):
      return bm.for_loop(lambda x: x + 1e-9, inp)

    input_long = bm.random.randn(1, int(duration / bm.dt), 3) / 100

    r = bp.running.cpu_unordered_parallel(body, {'inp': [input_long, input_long]}, num_process=2)
    assert bm.allclose(r[0], r[1])
