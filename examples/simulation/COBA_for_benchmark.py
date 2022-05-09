# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.math as bm

bp.math.set_platform('cpu')


class ExpCOBA(bp.dyn.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=1., delay=0., tau=8.0, E=0.,
               method='exp_auto'):
    super(ExpCOBA, self).__init__(pre=pre, post=post, conn=conn)
    self.check_pre_attrs('spike')
    self.check_post_attrs('input', 'V')

    # parameters
    self.E = E
    self.tau = tau
    self.delay = delay
    self.g_max = g_max
    self.pre2post = self.conn.require('pre2post')

    # variables
    self.g = bm.Variable(bm.zeros(self.post.num))

    # function
    self.integral = bp.odeint(lambda g, t: -g / self.tau, method=method)

  def update(self, t, dt):
    self.g.value = self.integral(self.g, t, dt=dt)
    self.g += bm.pre2post_event_sum(self.pre.spike, self.pre2post, self.post.num, self.g_max)
    self.post.input += self.g * (self.E - self.post.V)


class EINet(bp.dyn.Network):
  def __init__(self, scale=1.0, method='exp_auto'):
    # network size
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)

    # neurons
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)
    E = bp.models.LIF(num_exc, **pars, method=method)
    I = bp.models.LIF(num_inh, **pars, method=method)
    E.V[:] = bp.math.random.randn(num_exc) * 2 - 55.
    I.V[:] = bp.math.random.randn(num_inh) * 2 - 55.

    # synapses
    we = 0.6 / scale  # excitatory synaptic weight (voltage)
    wi = 6.7 / scale  # inhibitory synaptic weight
    E2E = ExpCOBA(E, E, bp.conn.FixedProb(prob=0.02), E=0., g_max=we, tau=5., method=method)
    E2I = ExpCOBA(E, I, bp.conn.FixedProb(prob=0.02), E=0., g_max=we, tau=5., method=method)
    I2E = ExpCOBA(I, E, bp.conn.FixedProb(prob=0.02), E=-80., g_max=wi, tau=10., method=method)
    I2I = ExpCOBA(I, I, bp.conn.FixedProb(prob=0.02), E=-80., g_max=wi, tau=10., method=method)

    super(EINet, self).__init__(E2E, E2I, I2E, I2I, E=E, I=I)


net = EINet(scale=10., method='euler')
# simulation
runner = bp.dyn.DSRunner(net,
                         # monitors=['E.spike'],
                         inputs=[('E.input', 20.), ('I.input', 20.)])
t = runner.run(10000.)
print(t)

# visualization
# bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], show=True)
