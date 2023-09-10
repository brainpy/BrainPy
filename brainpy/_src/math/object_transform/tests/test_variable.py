import brainpy.math as bm
import unittest


class TestVar(unittest.TestCase):
  def test1(self):
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




