import unittest
import brainpy.math as bm

try:
  from absl.testing import parameterized
except ImportError:
  pass


class UnitTestCase(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    bm.random.seed()
    self.rng = bm.random.default_rng()

  def __del__(self):
    bm.clear_buffer_memory()


