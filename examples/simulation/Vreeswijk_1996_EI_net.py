# -*- coding: utf-8 -*-
import brainpy as bp
import brainpy.math as bm

bm.set_platform('cpu')


class EINet(bp.dyn.Network):
  def __init__(self, num_exc, num_inh, prob, JE, JI):
    # neurons
    pars = dict(V_rest=-52., V_th=-50., V_reset=-60., tau=10., tau_ref=0.)
    E = bp.neurons.LIF(num_exc, **pars)
    I = bp.neurons.LIF(num_inh, **pars)
    E.V[:] = bm.random.random(num_exc) * (E.V_th - E.V_rest) + E.V_rest
    I.V[:] = bm.random.random(num_inh) * (E.V_th - E.V_rest) + E.V_rest

    # synapses
    E2E = bp.synapses.Exponential(E, E, bp.conn.FixedProb(prob), g_max=JE, tau=2.,
                                  output=bp.synouts.CUBA())
    E2I = bp.synapses.Exponential(E, I, bp.conn.FixedProb(prob), g_max=JE, tau=2.,
                                  output=bp.synouts.CUBA())
    I2E = bp.synapses.Exponential(I, E, bp.conn.FixedProb(prob), g_max=JI, tau=2.,
                                  output=bp.synouts.CUBA())
    I2I = bp.synapses.Exponential(I, I, bp.conn.FixedProb(prob), g_max=JI, tau=2.,
                                  output=bp.synouts.CUBA())

    super(EINet, self).__init__(E2E, E2I, I2E, I2I, E=E, I=I)


num_exc = 500
num_inh = 500
prob = 0.5

Ib = 3.
JE = 1 / bp.math.sqrt(prob * num_exc)
JI = -1 / bp.math.sqrt(prob * num_inh)

net = EINet(num_exc, num_inh, prob=prob, JE=JE, JI=JI)

runner = bp.dyn.DSRunner(net,
                         monitors=['E.spike'],
                         inputs=[('E.input', Ib), ('I.input', Ib)])
t = runner.run(1000.)

import matplotlib.pyplot as plt

fig, gs = bp.visualize.get_figure(4, 1, 2, 10)

fig.add_subplot(gs[:3, 0])
bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], xlim=(50, 950))

fig.add_subplot(gs[3, 0])
rates = bp.measure.firing_rate(runner.mon['E.spike'], 5.)
plt.plot(runner.mon.ts, rates)
plt.xlim(50, 950)
plt.show()
