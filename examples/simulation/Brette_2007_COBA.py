# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

bp.math.set_platform('cpu')


class EINet(bp.dyn.Network):
  def __init__(self, scale=1.0, method='exp_auto'):
    super(EINet, self).__init__()

    # network size
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)

    # neurons
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)
    self.E = bp.dyn.LIF(num_exc, **pars, method=method)
    self.I = bp.dyn.LIF(num_inh, **pars, method=method)
    self.E.V[:] = bm.random.randn(num_exc) * 2 - 55.
    self.I.V[:] = bm.random.randn(num_inh) * 2 - 55.

    # synapses
    we = 0.6 / scale  # excitatory synaptic weight (voltage)
    wi = 6.7 / scale  # inhibitory synaptic weight
    self.E2E = bp.dyn.ExpCOBA(self.E, self.E, bp.conn.FixedProb(0.02),
                              E=0., g_max=we, tau=5., method=method)
    self.E2I = bp.dyn.ExpCOBA(self.E, self.I, bp.conn.FixedProb(0.02),
                              E=0., g_max=we, tau=5., method=method)
    self.I2E = bp.dyn.ExpCOBA(self.I, self.E, bp.conn.FixedProb(0.02),
                              E=-80., g_max=wi, tau=10., method=method)
    self.I2I = bp.dyn.ExpCOBA(self.I, self.I, bp.conn.FixedProb(0.02),
                              E=-80., g_max=wi, tau=10., method=method)


net = EINet(scale=1., method='exp_auto')
# simulation
runner = bp.dyn.DSRunner(
  net,
  monitors={'E.spike': net.E.spike},
  inputs=[(net.E.input, 20.), (net.I.input, 20.)]
)
runner.run(100.)

# visualization
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)
