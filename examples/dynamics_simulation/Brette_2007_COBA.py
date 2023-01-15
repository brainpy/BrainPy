# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

bp.math.set_platform('cpu')


class EINet_V1(bp.Network):
  def __init__(self, scale=1.0, method='exp_auto'):
    super(EINet_V1, self).__init__()

    # network size
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)

    # neurons
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)
    self.E = bp.neurons.LIF(num_exc, **pars, method=method)
    self.I = bp.neurons.LIF(num_inh, **pars, method=method)
    self.E.V[:] = bm.random.randn(num_exc) * 2 - 55.
    self.I.V[:] = bm.random.randn(num_inh) * 2 - 55.

    # synapses
    we = 0.6 / scale  # excitatory synaptic weight (voltage)
    wi = 6.7 / scale  # inhibitory synaptic weight
    self.E2E = bp.synapses.Exponential(self.E, self.E, bp.conn.FixedProb(0.02),
                                       output=bp.synouts.COBA(E=0.), g_max=we,
                                       tau=5., method=method)
    self.E2I = bp.synapses.Exponential(self.E, self.I, bp.conn.FixedProb(0.02),
                                       output=bp.synouts.COBA(E=0.), g_max=we,
                                       tau=5., method=method)
    self.I2E = bp.synapses.Exponential(self.I, self.E, bp.conn.FixedProb(0.02),
                                       output=bp.synouts.COBA(E=-80.), g_max=wi,
                                       tau=10., method=method)
    self.I2I = bp.synapses.Exponential(self.I, self.I, bp.conn.FixedProb(0.02),
                                       output=bp.synouts.COBA(E=-80.), g_max=wi,
                                       tau=10., method=method)


def run_model_v1():
  net = EINet_V1(scale=1., method='exp_auto')
  # simulation
  runner = bp.DSRunner(
    net,
    monitors={'E.spike': net.E.spike},
    inputs=[(net.E.input, 20.), (net.I.input, 20.)]
  )
  runner.run(100.)

  # visualization
  bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)


class EINet_V2(bp.Network):
  def __init__(self, scale=1.0, method='exp_auto'):
    super(EINet_V2, self).__init__()

    # network size
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)

    # neurons
    self.N = bp.neurons.LIF(num_exc + num_inh,
                            V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.,
                            method=method, V_initializer=bp.initialize.Normal(-55., 2.))

    # synapses
    we = 0.6 / scale  # excitatory synaptic weight (voltage)
    wi = 6.7 / scale  # inhibitory synaptic weight
    self.Esyn = bp.synapses.Exponential(pre=self.N[:num_exc],
                                        post=self.N,
                                        conn=bp.connect.FixedProb(0.02),
                                        g_max=we, tau=5.,
                                        output=bp.synouts.COBA(E=0.),
                                        method=method)
    self.Isyn = bp.synapses.Exponential(pre=self.N[num_exc:],
                                        post=self.N,
                                        conn=bp.connect.FixedProb(0.02),
                                        g_max=wi, tau=10.,
                                        output=bp.synouts.COBA(E=-80.),
                                        method=method)


def run_model_v2():
  net = EINet_V2(scale=1., method='exp_auto')
  # simulation
  runner = bp.DSRunner(
    net,
    monitors={'spikes': net.N.spike},
    inputs=[(net.N.input, 20.)]
  )
  runner.run(100.)

  # visualization
  bp.visualize.raster_plot(runner.mon.ts, runner.mon['spikes'], show=True)


if __name__ == '__main__':
  run_model_v1()
  run_model_v2()
