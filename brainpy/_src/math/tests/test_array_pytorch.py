import unittest

import traceback
import brainpy.math as bm

class TestArrayPytorch(unittest.TestCase):
  def test_sin(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.sin(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_sin_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.sin_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_sin(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.sin(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_sin_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.sin_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_sinh(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.sinh(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_sinh_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.sinh_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_sinh(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.sinh(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_sinh_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.sinh_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_arcsin(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.arcsin(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_arcsin_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.arcsin_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_arcsin(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.arcsin(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_arcsin_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.arcsin_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_arcsinh(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.arcsinh(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_arcsinh_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.arcsinh_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_arcsinh(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.arcsinh(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_arcsinh_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.arcsinh_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_cos(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.cos(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_cos_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.cos_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_cos(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.cos(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_cos_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.cos_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_cosh(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.cosh(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_cosh_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.cosh_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_cosh(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.cosh(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_cosh_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.cosh_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_arccos(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.arccos(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_arccos_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.arccos_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_arccos(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.arccos(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_arccos_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.arccos_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_arccosh(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.arccosh(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_arccosh_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.arccosh_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_arccosh(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.arccosh(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_arccosh_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.arccosh_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_tan(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.tan(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_tan_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.tan_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_tan(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.tan(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_tan_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.tan_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_tanh(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.tanh(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_tanh_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.tanh_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_tanh(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.tanh(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_tanh_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.tanh_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_arctan(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.arctan(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_arctan_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.arctan_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_arctan(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.arctan(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_arctan_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.arctan_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_arctanh(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.arctanh(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_arctanh_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.arctanh_()
      print(a)
    except Exception as e:
      traceback.print_exc()

  def test_arctanh(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      b = bm.Array([-1, -1, -1], dtype=float)
      c = a.arctanh(out=b)
      print(c, b)
    except Exception as e:
      traceback.print_exc()

  def test_arctanh_(self):
    try:
      a = bm.Array([0, -2, 3], dtype=float)
      a.arctanh_()
      print(a)
    except Exception as e:
      traceback.print_exc()
