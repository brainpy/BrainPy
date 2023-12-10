
import brainpy.math as bm
from scipy.special import exprel

from unittest import TestCase


class Test_exprel(TestCase):
  def test1(self):
    for x in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
      print(f'{exprel(x)}, {bm.exprel(x)}, {exprel(x) - bm.exprel(x):.10f}')
      # self.assertEqual(exprel(x))

  def test2(self):
    bm.enable_x64()
    for x in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
      print(f'{exprel(x)}, {bm.exprel(x)}, {exprel(x) - bm.exprel(x):.10f}')
      # self.assertEqual(exprel(x))



