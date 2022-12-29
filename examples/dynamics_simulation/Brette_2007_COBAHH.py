# -*- coding: utf-8 -*-

import brainpy as bp
from brainpy.dyn import channels, synapses, synouts

bp.math.set_platform('cpu')


class HH(bp.CondNeuGroup):
  def __init__(self, size):
    super(HH, self).__init__(size, )
    self.INa = channels.INa_TM1991(size, g_max=100., V_sh=-63.)
    self.IK = channels.IK_TM1991(size, g_max=30., V_sh=-63.)
    self.IL = channels.IL(size, E=-60., g_max=0.05)


class EINet_v1(bp.Network):
  def __init__(self, scale=1.):
    super(EINet_v1, self).__init__()
    self.E = HH(int(3200 * scale))
    self.I = HH(int(800 * scale))
    prob = 0.02
    self.E2E = synapses.Exponential(self.E, self.E, bp.conn.FixedProb(prob),
                                    g_max=0.03 / scale, tau=5,
                                    output=synouts.COBA(E=0.))
    self.E2I = synapses.Exponential(self.E, self.I, bp.conn.FixedProb(prob),
                                    g_max=0.03 / scale, tau=5.,
                                    output=synouts.COBA(E=0.))
    self.I2E = synapses.Exponential(self.I, self.E, bp.conn.FixedProb(prob),
                                    g_max=0.335 / scale, tau=10.,
                                    output=synouts.COBA(E=-80))
    self.I2I = synapses.Exponential(self.I, self.I, bp.conn.FixedProb(prob),
                                    g_max=0.335 / scale, tau=10.,
                                    output=synouts.COBA(E=-80.))


class EINet_v2(bp.Network):
  def __init__(self, scale=1.):
    super(EINet_v2, self).__init__()

    prob = 0.02
    self.num_exc = int(3200 * scale)
    self.num_inh = int(800 * scale)

    self.N = HH(self.num_exc + self.num_inh)
    self.Esyn = bp.synapses.Exponential(self.N[:self.num_exc],
                                        self.N,
                                        bp.conn.FixedProb(prob),
                                        g_max=0.03 / scale, tau=5,
                                        output=synouts.COBA(E=0.))
    self.Isyn = bp.synapses.Exponential(self.N[self.num_exc:],
                                        self.N,
                                        bp.conn.FixedProb(prob),
                                        g_max=0.335 / scale, tau=10.,
                                        output=synouts.COBA(E=-80))


def run_ei_v1():
  net = EINet_v1(scale=1)
  runner = bp.DSRunner(net, monitors={'E.spike': net.E.spike})
  runner.run(100.)
  bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)


def run_ei_v2():
  net = EINet_v2(scale=1)
  runner = bp.DSRunner(net, monitors={'spikes': net.N.spike})
  runner.run(100.)
  bp.visualize.raster_plot(runner.mon.ts, runner.mon['spikes'], show=True)


if __name__ == '__main__':
  run_ei_v1()
  run_ei_v2()
