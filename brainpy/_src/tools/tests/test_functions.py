
import unittest

import brainpy as bp
import brainpy.math as bm


class TestFunction(unittest.TestCase):
  def test_compose(self):
    f = lambda a: a + 1
    g = lambda a: a * 10
    fun1 = bp.tools.compose(f, g)
    fun2 = bp.tools.pipe(g, f)

    arr = bm.random.randn(10)
    r1 = fun1(arr)
    r2 = fun2(arr)
    groundtruth = f(g(arr))
    self.assertTrue(bm.allclose(r1, r2))
    self.assertTrue(bm.allclose(r1, groundtruth))
    bm.clear_buffer_memory()



