import brainpy.math as bm
import brainunit as u
import jax.numpy as jnp
from functools import partial
import unittest


class TestVar(unittest.TestCase):
  def test_ndarray(self):
    class A(bm.BrainPyObject):
      def __init__(self):
        super().__init__()
        self.a = bm.Variable(1)
        self.f1 = bm.jit(self.f)
        self.f2 = bm.jit(self.ff)
        self.f3 = bm.jit(self.fff)

      def f(self):
        b = self.tracing_variable('b', bm.ones, (1,))
        self.a += (b * 2)
        return self.a.value

      def ff(self):
        self.b += 1.

      def fff(self):
        self.f()
        self.ff()
        self.b *= self.a
        return self.b.value

    print()
    f_jit = bm.jit(A().f)
    f_jit()
    self.assertTrue(len(f_jit._dyn_vars) == 2)

    print()
    a = A()
    temp = a.f1()
    print(temp)
    self.assertTrue(bm.all(a.f1() == 2.))
    self.assertTrue(len(a.f1._dyn_vars) == 2)
    print(a.f2())
    self.assertTrue(len(a.f2._dyn_vars) == 1)

    print()
    a = A()
    print()
    self.assertTrue(bm.allclose(a.f3(), 4.))
    self.assertTrue(len(a.f3._dyn_vars) == 2)

    bm.clear_buffer_memory()

  def test_state(self):
    class B(bm.BrainPyObject):
      def __init__(self):
        super().__init__()
        self.a = bm.Variable([0.,] * u.mV)
        self.f1 = bm.jit(self.f)
        self.f2 = bm.jit(self.ff)
        self.f3 = bm.jit(self.fff)

      def f(self):
        ones_fun = partial(u.math.ones,unit=u.mV)
        b = self.tracing_variable('b', ones_fun, (1,))
        self.a += (b * 2)
        return self.a.value

      def ff(self):
        self.b += 1. * u.mV

      def fff(self):
        self.f()
        self.ff()
        self.b *= self.a.value.mantissa
        return self.b.value

    print()
    f_jit = bm.jit(B().f)
    f_jit()
    self.assertTrue(len(f_jit._dyn_vars) == 2)

    print()
    b = B()
    self.assertTrue(u.math.all(b.f1() == [2.,] * u.mV))
    self.assertTrue(len(b.f1._dyn_vars) == 2)
    print(b.f2())
    self.assertTrue(len(b.f2._dyn_vars) == 1)

    print()
    b = B()
    print()
    self.assertTrue(u.math.allclose(b.f3(), 4. * u.mV))
    self.assertTrue(len(b.f3._dyn_vars) == 2)

    bm.clear_buffer_memory()




