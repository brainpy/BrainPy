import unittest

import pytest

import brainpy.version2.math as bm


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

        with pytest.raises(NotImplementedError):
            print()
            f_jit = bm.jit(A().f)
            f_jit()

            print()
            a = A()
