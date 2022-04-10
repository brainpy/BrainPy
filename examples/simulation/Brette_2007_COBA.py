# -*- coding: utf-8 -*-

import brainpy as bp

bp.math.set_platform('cpu')


class EINet(bp.dyn.Network):
  def __init__(self, scale=1.0, method='exp_auto'):
    # network size
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)

    # neurons
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)
    E = bp.dyn.LIF(num_exc, **pars, method=method)
    I = bp.dyn.LIF(num_inh, **pars, method=method)
    E.V[:] = bp.math.random.randn(num_exc) * 2 - 55.
    I.V[:] = bp.math.random.randn(num_inh) * 2 - 55.

    # synapses
    we = 0.6 / scale  # excitatory synaptic weight (voltage)
    wi = 6.7 / scale  # inhibitory synaptic weight
    E2E = bp.dyn.ExpCOBA(E, E, bp.conn.FixedProb(0.02),
                         E=0., g_max=we, tau=5., method=method)
    E2I = bp.dyn.ExpCOBA(E, I, bp.conn.FixedProb(0.02),
                         E=0., g_max=we, tau=5., method=method)
    I2E = bp.dyn.ExpCOBA(I, E, bp.conn.FixedProb(0.02),
                         E=-80., g_max=wi, tau=10., method=method)
    I2I = bp.dyn.ExpCOBA(I, I, bp.conn.FixedProb(0.02),
                         E=-80., g_max=wi, tau=10., method=method)

    super(EINet, self).__init__(E2E, E2I, I2E, I2I, E=E, I=I)


net = EINet(scale=1., method='exp_auto')
# simulation
runner = bp.dyn.DSRunner(net,
                         monitors=['E.spike'],
                         inputs=[('E.input', 20.), ('I.input', 20.)])
t = runner.run(100.)
print(t)

# visualization
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)
