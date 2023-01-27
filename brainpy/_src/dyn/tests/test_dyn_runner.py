# -*- coding: utf-8 -*-


import unittest
import brainpy as bp
import brainpy.math as bm


class TestDSRunner(unittest.TestCase):
  def test1(self):
    class ExampleDS(bp.DynamicalSystem):
      def __init__(self):
        super(ExampleDS, self).__init__()
        self.i = bm.Variable(bm.zeros(1))

      def update(self, tdi):
        self.i += 1

    ds = ExampleDS()
    runner = bp.DSRunner(ds, dt=1., monitors=['i'], progress_bar=False)
    runner.run(100.)

  def test_t_and_dt(self):
    class ExampleDS(bp.DynamicalSystem):
      def __init__(self):
        super(ExampleDS, self).__init__()
        self.i = bm.Variable(bm.zeros(1))

      def update(self, tdi):
        self.i += 1 * tdi.dt

    runner = bp.DSRunner(ExampleDS(), dt=1., monitors=['i'], progress_bar=False)
    runner.run(100.)

  def test_DSView(self):
    class EINet(bp.Network):
      def __init__(self, scale=1.0, method='exp_auto'):
        super(EINet, self).__init__()

        # network size
        num_exc = int(800 * scale)
        num_inh = int(200 * scale)

        # neurons
        pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)
        self.E = bp.neurons.LIF(num_exc, **pars, method=method)
        self.I = bp.neurons.LIF(num_inh, **pars, method=method)
        self.E.V[:] = bm.random.randn(num_exc) * 2 - 55.
        self.I.V[:] = bm.random.randn(num_inh) * 2 - 55.

        # synapses
        we = 0.6 / scale  # excitatory synaptic weight (voltage)
        wi = 6.7 / scale  # inhibitory synaptic weight
        self.E2E = bp.synapses.Exponential(self.E, self.E[:100], bp.conn.FixedProb(0.02),
                                           output=bp.synouts.COBA(E=0.), g_max=we,
                                           tau=5., method=method)
        self.E2I = bp.synapses.Exponential(self.E, self.I[:100], bp.conn.FixedProb(0.02),
                                           output=bp.synouts.COBA(E=0.), g_max=we,
                                           tau=5., method=method)
        self.I2E = bp.synapses.Exponential(self.I, self.E[:100], bp.conn.FixedProb(0.02),
                                           output=bp.synouts.COBA(E=-80.), g_max=wi,
                                           tau=10., method=method)
        self.I2I = bp.synapses.Exponential(self.I, self.I[:100], bp.conn.FixedProb(0.02),
                                           output=bp.synouts.COBA(E=-80.), g_max=wi,
                                           tau=10., method=method)

    net = EINet(scale=1., method='exp_auto')
    # with JIT
    runner = bp.DSRunner(net, monitors={'E.spike': net.E.spike},
                         inputs=[(net.E.input, 20.), (net.I.input, 20.)]).run(1.)

    # without JIT
    runner = bp.DSRunner(net, monitors={'E.spike': net.E.spike},
                         inputs=[(net.E.input, 20.), (net.I.input, 20.)],
                         jit=False).run(0.2)






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
