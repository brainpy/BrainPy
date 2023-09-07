import unittest
import brainpy.math as bm
import brainpy as bp


class TestDiffEncoder(unittest.TestCase):
  def test_delta(self):
    a = bm.array([1, 2, 2.9, 3, 3.9])
    encoder = bp.encoding.DiffEncoder(threshold=1)
    r = encoder.multi_steps(a)
    excepted = bm.asarray([1., 1., 0., 0., 0.])
    self.assertTrue(bm.allclose(r, excepted))

    encoder = bp.encoding.DiffEncoder(threshold=1, padding=True)
    r = encoder.multi_steps(a)
    excepted = bm.asarray([0., 1., 0., 0., 0.])
    self.assertTrue(bm.allclose(r, excepted))

    bm.clear_buffer_memory()

  def test_delta_off_spike(self):
    b = bm.array([1, 2, 0, 2, 2.9])
    encoder = bp.encoding.DiffEncoder(threshold=1, off_spike=True)
    r = encoder.multi_steps(b)
    excepted = bm.asarray([1., 1., -1., 1., 0.])
    self.assertTrue(bm.allclose(r, excepted))

    encoder = bp.encoding.DiffEncoder(threshold=1, padding=True, off_spike=True)
    r = encoder.multi_steps(b)
    excepted = bm.asarray([0., 1., -1., 1., 0.])
    self.assertTrue(bm.allclose(r, excepted))

    bm.clear_buffer_memory()


class TestLatencyEncoder(unittest.TestCase):
  def test_latency(self):
    a = bm.array([0.02, 0.5, 1])
    encoder = bp.encoding.LatencyEncoder(method='linear')

    r = encoder.multi_steps(a, n_time=0.5)
    excepted = bm.asarray(
      [[0., 0., 1.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 1., 0.],
       ]
    )
    self.assertTrue(bm.allclose(r, excepted))

    r = encoder.multi_steps(a, n_time=1.0)
    excepted = bm.asarray(
      [[0., 0., 1.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 1., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [1., 0., 0.],
       ]
    )
    self.assertTrue(bm.allclose(r, excepted))

    encoder = bp.encoding.LatencyEncoder(method='linear', normalize=True)
    r = encoder.multi_steps(a, n_time=0.5)
    excepted = bm.asarray(
      [[0., 0., 1.],
       [0., 0., 0.],
       [0., 1., 0.],
       [0., 0., 0.],
       [1., 0., 0.],
       ]
    )
    self.assertTrue(bm.allclose(r, excepted))

