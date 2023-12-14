import unittest

import brainpy as bp


def test1():
  class A(bp.DynamicalSystem):
    def update(self, x=None):
      # print(tdi)
      print(x)

  A()({}, 10.)


def test2():
  class B(bp.DynamicalSystem):
    def update(self, tdi, x=None):
      print(tdi)
      print(x)

  B()({}, 10.)
  B()(10.)


def test3():
  class A(bp.DynamicalSystem):
    def update(self, x=None):
      # print(tdi)
      print('A:', x)

  class B(A):
    def update(self, tdi, x=None):
      print('B:', tdi, x)
      super().update(x)

  B()(dict(), 1.)
  B()(1.)


class TestResetLevelDecorator(unittest.TestCase):
  _max_level = 10  # Define the maximum level for testing purposes

  @bp.reset_level(5)
  def test_function_with_reset_level_5(self):
    self.assertEqual(self.test_function_with_reset_level_5.reset_level, 5)

  def test1(self):
    with self.assertRaises(ValueError):
      @bp.reset_level(12)  # This should raise a ValueError
      def test_function_with_invalid_reset_level(self):
          pass  # Call the function here to trigger the ValueError

  @bp.reset_level(-3)
  def test_function_with_negative_reset_level(self):
    self.assertEqual(self.test_function_with_negative_reset_level.reset_level, self._max_level - 3)
