import unittest

import matplotlib.pyplot as plt

import brainpy as bp
import brainpy.math as bm

show = False


class TestDualExpon(unittest.TestCase):
  def test_dual_expon(self):
    # bm.set(dt=0.01)

    class Net(bp.DynSysGroup):
      def __init__(self, tau_r, tau_d, n_spk):
        super().__init__()

        self.inp = bp.dyn.SpikeTimeGroup(1, bm.zeros(n_spk, dtype=int), bm.linspace(2., 100., n_spk))
        self.proj = bp.dyn.DualExpon(1, tau_rise=tau_r, tau_decay=tau_d)

      def update(self):
        self.proj(self.inp())
        return self.proj.h.value, self.proj.g.value

    for tau_r, tau_d in [(1., 10.), (10., 100.)]:
      for n_spk in [1, 10, 100]:
        net = Net(tau_r, tau_d, n_spk)
        indices = bm.as_numpy(bm.arange(1000))
        hs, gs = bm.for_loop(net.step_run, indices, progress_bar=True)

        bp.visualize.line_plot(indices * bm.get_dt(), hs, legend='h')
        bp.visualize.line_plot(indices * bm.get_dt(), gs, legend='g', show=show)
    plt.close('all')


  def test_dual_expon_v2(self):
    class Net(bp.DynSysGroup):
      def __init__(self, tau_r, tau_d, n_spk):
        super().__init__()

        self.inp = bp.dyn.SpikeTimeGroup(1, bm.zeros(n_spk, dtype=int), bm.linspace(2., 100., n_spk))
        self.syn = bp.dyn.DualExponV2(1, tau_rise=tau_r, tau_decay=tau_d)

      def update(self):
        return self.syn(self.inp())

    for tau_r, tau_d in [(1., 10.), (5., 50.), (10., 100.)]:
      for n_spk in [1, 10, 100]:
        net = Net(tau_r, tau_d, n_spk)
        indices = bm.as_numpy(bm.arange(1000))
        gs = bm.for_loop(net.step_run, indices, progress_bar=True)

        bp.visualize.line_plot(indices * bm.get_dt(), gs, legend='g', show=show)

    plt.close('all')

class TestAlpha(unittest.TestCase):

  def test_v1(self):
    class Net(bp.DynSysGroup):
      def __init__(self, tau, n_spk):
        super().__init__()

        self.inp = bp.dyn.SpikeTimeGroup(1, bm.zeros(n_spk, dtype=int), bm.linspace(2., 100., n_spk))
        self.neu = bp.dyn.LifRef(1)
        self.proj = bp.dyn.FullProjAlignPreDS(self.inp, None,
                                              bp.dyn.Alpha(1, tau_decay=tau),
                                              bp.dnn.AllToAll(1, 1, 1.),
                                              bp.dyn.CUBA(), self.neu)

      def update(self):
        self.inp()
        self.proj()
        self.neu()
        return self.proj.syn.h.value, self.proj.syn.g.value

    for tau in [10.]:
      for n_spk in [1, 10, 50]:
        net = Net(tau=tau, n_spk=n_spk)
        indices = bm.as_numpy(bm.arange(1000))
        hs, gs = bm.for_loop(net.step_run, indices, progress_bar=True)

        bp.visualize.line_plot(indices * bm.get_dt(), hs, legend='h')
        bp.visualize.line_plot(indices * bm.get_dt(), gs, legend='g', show=show)

    plt.close('all')
