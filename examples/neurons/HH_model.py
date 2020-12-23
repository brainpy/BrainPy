# -*- coding: utf-8 -*-

import brainpy as bp
import brainpy.numpy as np


def define_hh(noise=0., E_Na=50., g_Na=120., E_K=-77., g_K=36., E_Leak=-54.387,
              g_Leak=0.03, C=1.0, Vth=20.):
    """The Hodgkin–Huxley neuron model.

    Parameters
    ----------
    noise : float
        The noise fluctuation.
    E_Na : float
    g_Na : float
    E_K : float
    g_K : float
    E_Leak : float
    g_Leak : float
    C : float
    Vth : float

    """

    ST = bp.types.NeuState(
        {'V': -65., 'm': 0., 'h': 0., 'n': 0., 'sp': 0., 'inp': 0.},
        help='Hodgkin–Huxley neuron state.\n'
             '"V" denotes membrane potential.\n'
             '"n" denotes potassium channel activation probability.\n'
             '"m" denotes sodium channel activation probability.\n'
             '"h" denotes sodium channel inactivation probability.\n'
             '"sp" denotes spiking state.\n'
             '"inp" denotes synaptic input.\n'
    )

    @bp.integrate
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m

    @bp.integrate
    def int_h(h, t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        return alpha * (1 - h) - beta * h

    @bp.integrate
    def int_n(n, t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        return alpha * (1 - n) - beta * n

    @bp.integrate
    def int_V(V, t, m, h, n, Isyn):
        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + Isyn) / C
        return dvdt, noise / C

    def update(ST, _t):
        m = np.clip(int_m(ST['m'], _t, ST['V']), 0., 1.)
        h = np.clip(int_h(ST['h'], _t, ST['V']), 0., 1.)
        n = np.clip(int_n(ST['n'], _t, ST['V']), 0., 1.)
        V = int_V(ST['V'], _t, m, h, n, ST['inp'])
        sp = np.logical_and(ST['V'] < Vth, V >= Vth)
        ST['sp'] = sp
        ST['V'] = V
        ST['m'] = m
        ST['h'] = h
        ST['n'] = n
        ST['inp'] = 0.

    return bp.NeuType(name='HH_neuron',
                      ST=ST,
                      steps=update,
                      mode='vector')


if __name__ == '__main__':
    bp.profile.set(jit=False, dt=0.02, numerical_method='exponential')

    HH = define_hh(noise=0.)
    neu = bp.NeuGroup(HH, geometry=(100,), monitors=['sp', 'V', 'm', 'h', 'n'])
    neu.ST['V'] = np.random.random(100) * 20 + -75  # set initial variable state
    neu.pars['g_K'] = np.random.random(100) * 2 + 35  # update parameters

    neu.run(duration=50., inputs=['ST.inp', 10.], report=True)

    ts = neu.mon.ts
    fig, gs = bp.visualize.get_figure(2, 1, 3, 8)
    xlim = (-0.1, 50.1)

    ax = fig.add_subplot(gs[0, 0])
    bp.visualize.line_plot(ts, neu.mon.V, ax=ax, ylabel='Membrane potential', xlim=xlim)

    ax = fig.add_subplot(gs[1, 0])
    bp.visualize.line_plot(ts, neu.mon.m, ax=ax, legend='m')
    bp.visualize.line_plot(ts, neu.mon.n, ax=ax, legend='n')
    bp.visualize.line_plot(ts, neu.mon.h, ax=ax, legend='h', xlim=xlim, show=True)
