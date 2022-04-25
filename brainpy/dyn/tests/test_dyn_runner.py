# -*- coding: utf-8 -*-


import unittest
import brainpy as bp
import brainpy.math as bm


class TestDSRunner(unittest.TestCase):
  def test1(self):
    class ExampleDS(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(ExampleDS, self).__init__()
        self.i = bm.Variable(bm.zeros(1))

      def update(self, _t, _dt):
        self.i += 1

    ds = ExampleDS()
    runner = bp.dyn.DSRunner(ds, dt=1., monitors=['i'], progress_bar=False)
    runner.run(100.)

  def test_t_and_dt(self):
    class ExampleDS(bp.dyn.DynamicalSystem):
      def __init__(self):
        super(ExampleDS, self).__init__()
        self.i = bm.Variable(bm.zeros(1))

      def update(self, t, dt):
        self.i += 1 * dt

    runner = bp.dyn.DSRunner(ExampleDS(), dt=1., monitors=['i'], progress_bar=False)
    runner.run(100.)

# class TestMonitor(TestCase):
#   def test_1d_array(self):
#     try1 = TryGroup(monitors=['a'])
#     try1.a = np.ones(1)
#     try1.run(100.)
#
#     assert np.ndim(try1.mon.a) == 2 and np.shape(try1.mon.a)[1] == 1
#     assert np.allclose(np.arange(2, 1002).reshape((-1, 1)), try1.mon.a)
#
#   def test_2d_array():
#     set(dt=0.1)
#     try1 = TryGroup(monitors=['a'])
#     try1.a = np.ones((2, 2))
#     try1.run(100.)
#
#     assert np.ndim(try1.mon.a) == 2 and np.shape(try1.mon.a)[1] == 4
#     series = np.arange(2, 1002).reshape((-1, 1))
#     series = np.repeat(series, 4, axis=1)
#     assert np.allclose(series, try1.mon.a)
#
#   def test_monitor_with_every():
#     set(dt=0.1)
#
#     # try1: 2d array
#     try1 = TryGroup(monitors=Monitor(variables=['a'], every=[1.]))
#     try1.run(100.)
#     assert np.ndim(try1.mon.a) == 2 and np.shape(try1.mon.a)[1] == 4
#     series = np.arange(2, 1002, 1. / 0.1).reshape((-1, 1))
#     series = np.repeat(series, 4, axis=1)
#     assert np.allclose(series, try1.mon.a)
#
#     # try2: 1d array
#     try2 = TryGroup(monitors=Monitor(variables=['a'], every=[1.]))
#     try2.a = np.array([1., 1.])
#     try2.run(100.)
#     assert np.ndim(try2.mon.a) == 2 and np.shape(try2.mon.a)[1] == 2
#     series = np.arange(2, 1002, 1. / 0.1).reshape((-1, 1))
#     series = np.repeat(series, 2, axis=1)
#     assert np.allclose(series, try2.mon.a)
#
#     # try2: scalar
#     try3 = TryGroup(monitors=Monitor(variables=['a'], every=[1.]))
#     try3.a = 1.
#     try3.run(100.)
#     assert np.ndim(try3.mon.a) == 2 and np.shape(try3.mon.a)[1] == 1
#     series = np.arange(2, 1002, 1. / 0.1).reshape((-1, 1))
#     assert np.allclose(series, try3.mon.a)
