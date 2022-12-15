# -*- coding: utf-8 -*-

import unittest

import matplotlib.pyplot as plt

import brainpy as bp
import brainpy.math as bm

bm.set_platform('cpu')


def abs_eval(events, indices, indptr, post_val, values):
  return [post_val]


def event_sum_op(outs, ins):
  events, indices, indptr, post, values = ins
  v = values[()]
  outs, = outs
  outs.fill(0)
  for i in range(len(events)):
    if events[i]:
      for j in range(indptr[i], indptr[i + 1]):
        index = indices[j]
        outs[index] += v


event_sum2 = bm.XLACustomOp(name='event_sum2', cpu_func=event_sum_op, eval_shape=abs_eval)


class ExponentialSyn(bp.dyn.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=1., delay=0., tau=8.0, E=0.,
               method='exp_auto'):
    super(ExponentialSyn, self).__init__(pre=pre, post=post, conn=conn)
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

  def update(self, tdi):
    self.g.value = self.integral(self.g, tdi['t'], dt=tdi['dt'])
    self.g += bm.pre2post_event_sum(self.pre.spike, self.pre2post, self.post.num, self.g_max)
    self.post.input += self.g * (self.E - self.post.V)


class ExponentialSyn3(bp.dyn.TwoEndConn):
  def __init__(self, pre, post, conn, g_max=1., delay=0., tau=8.0, E=0.,
               method='exp_auto'):
    super(ExponentialSyn3, self).__init__(pre=pre, post=post, conn=conn)
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

  def update(self, tdi):
    self.g.value = self.integral(self.g, tdi['t'], tdi['dt'])
    # Customized operator
    # ------------------------------------------------------------------------------------------------------------
    post_val = bm.zeros(self.post.num)
    r = event_sum2(self.pre.spike, self.pre2post[0], self.pre2post[1], post_val, self.g_max)
    self.g += r[0]
    # ------------------------------------------------------------------------------------------------------------
    self.post.input += self.g * (self.E - self.post.V)


class EINet(bp.dyn.Network):
  def __init__(self, syn_class, scale=1.0, method='exp_auto', ):
    super(EINet, self).__init__()

    # network size
    num_exc = int(3200 * scale)
    num_inh = int(800 * scale)

    # neurons
    pars = dict(V_rest=-60., V_th=-50., V_reset=-60., tau=20., tau_ref=5.)
    self.E = bp.neurons.LIF(num_exc, **pars, method=method)
    self.I = bp.neurons.LIF(num_inh, **pars, method=method)
    rng = bm.random.RandomState()
    self.E.V[:] = rng.randn(num_exc) * 2 - 55.
    self.I.V[:] = rng.randn(num_inh) * 2 - 55.

    # synapses
    we = 0.6 / scale  # excitatory synaptic weight (voltage)
    wi = 6.7 / scale  # inhibitory synaptic weight
    self.E2E = syn_class(self.E, self.E, bp.conn.FixedProb(0.02), E=0., g_max=we, tau=5., method=method)
    self.E2I = syn_class(self.E, self.I, bp.conn.FixedProb(0.02), E=0., g_max=we, tau=5., method=method)
    self.I2E = syn_class(self.I, self.E, bp.conn.FixedProb(0.02), E=-80., g_max=wi, tau=10., method=method)
    self.I2I = syn_class(self.I, self.I, bp.conn.FixedProb(0.02), E=-80., g_max=wi, tau=10., method=method)


class TestOpRegister(unittest.TestCase):
  def test_op(self):

    fig, gs = bp.visualize.get_figure(1, 1, 4, 5)

    net = EINet(ExponentialSyn, scale=1., method='euler')
    runner = bp.dyn.DSRunner(
      net,
      inputs=[(net.E.input, 20.),
              (net.I.input, 20.)],
      monitors={'E.spike': net.E.spike},
    )
    t, _ = runner.run(100., eval_time=True)
    print(t)
    ax = fig.add_subplot(gs[0, 0])
    bp.visualize.raster_plot(runner.mon.ts, runner.mon['E.spike'], ax=ax)

    net3 = EINet(ExponentialSyn3, scale=1., method='euler')
    runner3 = bp.dyn.DSRunner(
      net3,
      inputs=[(net3.E.input, 20.),
              (net3.I.input, 20.)],
      monitors={'E.spike': net3.E.spike},
    )
    t, _ = runner3.run(100., eval_time=True)
    print(t)
    ax = fig.add_subplot(gs[0, 1])
    bp.visualize.raster_plot(runner3.mon.ts, runner3.mon['E.spike'], ax=ax, show=True)

    # clear
    bm.clear_buffer_memory()
    plt.close()
