import unittest
import brainpy.math as bm
import numpy as np

try:
  from absl.testing import parameterized
except ImportError:
  pass


class UnitTestCase(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    bm.random.seed(np.random.randint(0, 100000))
    self.rng = bm.random.RandomState(np.random.randint(0, 100000))

  def __del__(self):
    bm.clear_buffer_memory()


