
import brainpy as bp
import unittest


class TestVarDelay(unittest.TestCase):
  def test_delay1(self):
    bp.math.random.seed()
    a = bp.math.Variable((10, 20))
    delay = bp.VarDelay(a,)
    delay.register_entry('a', 1.)
    delay.register_entry('b', 2.)
    delay.register_entry('c', None)
    with self.assertRaises(KeyError):
      delay.register_entry('c', 10.)
    bp.math.clear_buffer_memory()






