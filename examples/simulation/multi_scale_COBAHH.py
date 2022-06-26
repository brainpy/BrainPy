# -*- coding: utf-8 -*-

import numpy as np

import brainpy as bp
import brainpy.math as bm
from brainpy.dyn.channels import INa_TM1991, IL
from brainpy.dyn.synapses import Exponential
from brainpy.dyn.synouts import COBA


class IK2(bp.dyn.channels.IK_p4_markov):
  def __init__(self, size, E=-90., g_max=10., phi=1., V_sh=-50.):
    super(IK2, self).__init__(size, g_max=g_max, phi=phi, E=E)
    self.V_sh = V_sh

  def f_p_alpha(self, V):
    tmp = V - self.V_sh - 15.
    return 0.032 * tmp / (1. - bm.exp(-tmp / 5.))

  def f_p_beta(self, V):
    return 0.5 * bm.exp(-(V - self.V_sh - 10.) / 40.)


class IK(bp.dyn.Channel):
  def __init__(self, size, E=-90., g_max=10., phi=1., V_sh=-50.):
    super(IK, self).__init__(size)
    self.g_max, self.E, self.V_sh, self.phi = g_max, E, V_sh, phi
    self.p = bm.Variable(bm.zeros(size))
    self.integral = bp.odeint(self.dp, method='exp_euler')

  def dp(self, p, t, V):
    tmp = V - self.V_sh - 15.
    alpha = 0.032 * tmp / (1. - bm.exp(-tmp / 5.))
    beta = 0.5 * bm.exp(-(V - self.V_sh - 10.) / 40.)
    return self.phi * (alpha * (1. - p) - beta * p)

  def update(self, tdi, V):
    self.p.value = self.integral(self.p, tdi.t, V, dt=tdi.dt)

  def current(self, V):
    return self.g_max * self.p ** 4 * (self.E - V)


class HH(bp.dyn.CondNeuGroup):
  def __init__(self, size):
    super(HH, self).__init__(size, )
    self.INa = INa_TM1991(size, g_max=100., V_sh=-63.)
    self.IK = IK(size, g_max=30., V_sh=-63.)
    self.IL = IL(size, E=-60., g_max=0.05)


class Network(bp.dyn.Network):
  def __init__(self, num_E, num_I, ):
    super(Network, self).__init__()
    self.E = HH(num_E)
    self.I = HH(num_I)
    self.E2E = Exponential(self.E, self.E, bp.conn.FixedProb(0.02),
                           g_max=0.03, tau=5, output=COBA(E=0.))
    self.E2I = Exponential(self.E, self.I, bp.conn.FixedProb(0.02),
                           g_max=0.03, tau=5., output=COBA(E=0.))
    self.I2E = Exponential(self.I, self.E, bp.conn.FixedProb(0.02),
                           g_max=0.335, tau=10., output=COBA(E=-80))
    self.I2I = Exponential(self.I, self.I, bp.conn.FixedProb(0.02),
                           g_max=0.335, tau=10., output=COBA(E=-80.))


class Projection(bp.dyn.DynamicalSystem):
  def __init__(self, pre, post, delay, conn, g_max=0.03, tau=5.):
    super(Projection, self).__init__()
    self.pre = pre
    self.post = post

    g_max = conn * g_max
    self.E2E = Exponential(pre.E, post.E, bp.conn.FixedProb(0.02),
                           delay_step=delay, g_max=g_max, tau=tau,
                           output=COBA(0.))
    self.E2I = Exponential(pre.E, post.I, bp.conn.FixedProb(0.02),
                           delay_step=delay, g_max=g_max, tau=tau,
                           output=COBA(0.))

  def update(self, tdi):
    self.E2E.update(tdi)
    self.E2I.update(tdi)


class Circuit(bp.dyn.Network):
  def __init__(self, conn, delay):
    super(Circuit, self).__init__()

    num_area = conn.shape[0]
    self.areas = [Network(3200, 800) for _ in range(num_area)]
    self.projections = []
    for i in range(num_area):
      for j in range(num_area):
        if i != j:
          proj = Projection(self.areas[j], self.areas[i],
                            delay=delay[i, j], conn=conn[i, j])
          self.projections.append(proj)
    self.register_implicit_nodes(self.projections, self.areas)


bp.math.random.seed(1234)

data = np.load('./data/visual_conn.npz')
conn_data = data['conn']
delay_data = (data['delay'] / bm.get_dt()).astype(int)

circuit = Circuit(conn_data, delay_data)
f1 = lambda tdi: bm.concatenate([area.E.spike for area in circuit.areas])
f2 = lambda tdi: bm.concatenate([area.I.spike for area in circuit.areas])
I, duration = bp.inputs.section_input([0, 0.8, 0.], [50., 50., 100.], return_length=True)
runner = bp.dyn.DSRunner(
  circuit,
  monitors={'K.p': circuit.areas[0].E.IK.p,
            'A0.V': (circuit.areas[0].E.V,),
            'A0.spike': circuit.areas[0].E.spike},
  fun_monitors={'exc.spike': f1, 'inh.spike': f2},
  # inputs=[circuit.areas[0].E.input, I, 'iter']
)
runner.run(duration)

fig, gs = bp.visualize.get_figure(2, 1, 4, 10)
fig.add_subplot(gs[0, 0])
bp.visualize.raster_plot(runner.mon['ts'], runner.mon.get('exc.spike'))
fig.add_subplot(gs[1, 0])
bp.visualize.raster_plot(runner.mon['ts'], runner.mon.get('inh.spike'), show=True)

import seaborn as sns

sns.set_theme(font_scale=1.5)

fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(runner.mon['ts'], runner.mon['K.p'], show=True, plot_ids=(4, 5, 1))

fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
fig.add_subplot(gs[0, 0])
bp.visualize.line_plot(runner.mon['ts'], runner.mon['A0.V'], show=True, plot_ids=(4, 5, 1))

fig, gs = bp.visualize.get_figure(1, 1, 4.5, 6)
fig.add_subplot(gs[0, 0])
bp.visualize.raster_plot(runner.mon['ts'], runner.mon['A0.spike'], show=True)
