import unittest

import brainpy.math as bm


class TestDefaults(unittest.TestCase):
  def test_dt(self):
    with bm.environment(dt=1.0):
      self.assertEqual(bm.dt, 1.0)
      self.assertEqual(bm.get_dt(), 1.0)

  def test_bool(self):
    with bm.environment(bool_=bm.int32):
      self.assertTrue(bm.bool_ == bm.int32)
      self.assertTrue(bm.get_bool() == bm.int32)

  def test_int(self):
    with bm.environment(int_=bm.int32):
      self.assertTrue(bm.int == bm.int32)
      self.assertTrue(bm.get_int() == bm.int32)

  def test_float(self):
    with bm.environment(float_=bm.float32):
      self.assertTrue(bm.float_ == bm.float32)
      self.assertTrue(bm.get_float() == bm.float32)

  def test_complex(self):
    with bm.environment(complex_=bm.complex64):
      self.assertTrue(bm.complex_ == bm.complex64)
      self.assertTrue(bm.get_complex() == bm.complex64)

  def test_mode(self):
    mode = bm.TrainingMode()
    with bm.environment(mode=mode):
      self.assertTrue(bm.mode == mode)
      self.assertTrue(bm.get_mode() == mode)
