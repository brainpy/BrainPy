# -*- coding: utf-8 -*-

"""
Implementation of the paper:

- Wang, Xiao-Jing, and György Buzsáki. “Gamma oscillation by
  synaptic inhibition in a hippocampal interneuronal network
  model.” Journal of neuroscience 16.20 (1996): 6402-6413.

"""

import brainpy as bp
import numpy as np

bp.profile.set(jit=True, dt=0.04, numerical_method='exponential')

# HH neuron model #
# --------------- #


V_th = 0.
C = 1.0
gLeak = 0.1
ELeak = -65
gNa = 35.
ENa = 55.
gK = 9.
EK = -90.
phi = 5.0

HH_ST = bp.types.NeuState({'V': -55., 'h': 0., 'n': 0., 'sp': 0., 'inp': 0.})


@bp.integrate
def int_h(h, t, V):
    alpha = 0.07 * np.exp(-(V + 58) / 20)
    beta = 1 / (np.exp(-0.1 * (V + 28)) + 1)
    dhdt = alpha * (1 - h) - beta * h
    return phi * dhdt


@bp.integrate
def int_n(n, t, V):
    alpha = -0.01 * (V + 34) / (np.exp(-0.1 * (V + 34)) - 1)
    beta = 0.125 * np.exp(-(V + 44) / 80)
    dndt = alpha * (1 - n) - beta * n
    return phi * dndt


@bp.integrate
def int_V(V, t, h, n, Isyn):
    m_alpha = -0.1 * (V + 35) / (np.exp(-0.1 * (V + 35)) - 1)
    m_beta = 4 * np.exp(-(V + 60) / 18)
    m = m_alpha / (m_alpha + m_beta)
    INa = gNa * m ** 3 * h * (V - ENa)
    IK = gK * n ** 4 * (V - EK)
    IL = gLeak * (V - ELeak)
    dvdt = (- INa - IK - IL + Isyn) / C
    return dvdt


def update(ST, _t):
    h = int_h(ST['h'], _t, ST['V'])
    n = int_n(ST['n'], _t, ST['V'])
    V = int_V(ST['V'], _t, ST['h'], ST['n'], ST['inp'])
    sp = np.logical_and(ST['V'] < V_th, V >= V_th)
    ST['sp'] = sp
    ST['V'] = V
    ST['h'] = h
    ST['n'] = n
    ST['inp'] = 0.


HH = bp.NeuType('HH_neuron', ST=HH_ST, steps=update)

# GABAa #
# ----- #

g_max = 0.1
E = -75.
alpha = 12.
beta = 0.1

requires = dict(
    pre=bp.types.NeuState(['V']),
    post=bp.types.NeuState(['V', 'inp']),
)


@bp.integrate
def int_s(s, t, TT):
    return alpha * TT * (1 - s) - beta * s


def update(ST, _t, pre, pre2syn):
    pre_above_th = pre['V'] - V_th
    for pre_id, syn_ids in enumerate(pre2syn):
        ST['pre_above_th'][syn_ids] = pre_above_th[pre_id]
    T = 1 / (1 + np.exp(-ST['pre_above_th'] / 2))
    s = int_s(ST['s'], _t, T)
    ST['s'] = s
    ST['g'] = g_max * s


def output(ST, post, post_slice_syn):
    num_post = post_slice_syn.shape[0]
    post_cond = np.empty(num_post, dtype=np.float_)
    for post_id in range(num_post):
        pos = post_slice_syn[post_id]
        post_cond[post_id] = np.sum(ST['g'][pos[0]: pos[1]])
    post['inp'] -= post_cond * (post['V'] - E)


GABAa = bp.SynType('GABAa',
                   ST=bp.types.SynState(['g', 's', 'pre_above_th']),
                   requires=requires,
                   steps=(update, output))

if __name__ == '__main__':
    num = 100
    v_init = -70. + np.random.random(num) * 20
    h_alpha = 0.07 * np.exp(-(v_init + 58) / 20)
    h_beta = 1 / (np.exp(-0.1 * (v_init + 28)) + 1)
    h_init = h_alpha / (h_alpha + h_beta)
    n_alpha = -0.01 * (v_init + 34) / (np.exp(-0.1 * (v_init + 34)) - 1)
    n_beta = 0.125 * np.exp(-(v_init + 44) / 80)
    n_init = n_alpha / (n_alpha + n_beta)

    num = 100
    neu = bp.NeuGroup(HH, geometry=num, monitors=['sp', 'V'])
    neu.ST['V'] = -70. + np.random.random(num) * 20
    neu.ST['h'] = h_init
    neu.ST['n'] = n_init

    syn = bp.SynConn(GABAa, pre_group=neu, post_group=neu,
                     conn=bp.connect.All2All(include_self=False),
                     monitors=['s', 'g'])
    syn.pars['g_max'] = 0.1 / num

    net = bp.Network(neu, syn)
    net.run(duration=500., inputs=[neu, 'ST.inp', 1.2], report=True, report_percent=0.2)

    fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
    xlim = (net.t_start - 0.1, net.t_end + 0.1)

    fig.add_subplot(gs[0, 0])
    bp.visualize.line_plot(net.ts, neu.mon.V, xlim=xlim, ylabel='Membrane potential (N0)')

    fig.add_subplot(gs[1, 0])
    bp.visualize.raster_plot(net.ts, neu.mon.sp, xlim=xlim, show=True)
