# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import brainpy as nb
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

    Returns
    -------
    return_dict : dict
        The necessary variables.
    """

    ST = nb.types.NeuState(
        {'V': -65., 'm': 0., 'h': 0., 'n': 0., 'sp': 0., 'inp': 0.},
        help='Hodgkin–Huxley neuron state.\n'
             '"V" denotes membrane potential.\n'
             '"n" denotes potassium channel activation probability.\n'
             '"m" denotes sodium channel activation probability.\n'
             '"h" denotes sodium channel inactivation probability.\n'
             '"sp" denotes spiking state.\n'
             '"inp" denotes synaptic input.\n'
    )

    @nb.integrate
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m

    @nb.integrate
    def int_h(h, t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        return alpha * (1 - h) - beta * h

    @nb.integrate
    def int_n(n, t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        return alpha * (1 - n) - beta * n

    @nb.integrate(noise=noise / C)
    def int_V(V, t, m, h, n, Isyn):
        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + Isyn) / C
        return dvdt

    def update(ST, _t_):
        m = np.clip(int_m(ST['m'], _t_, ST['V']), 0., 1.)
        h = np.clip(int_h(ST['h'], _t_, ST['V']), 0., 1.)
        n = np.clip(int_n(ST['n'], _t_, ST['V']), 0., 1.)
        V = int_V(ST['V'], _t_, m, h, n, ST['inp'])
        sp = np.logical_and(ST['V'] < Vth, V >= Vth)
        ST['sp'] = sp
        ST['V'] = V
        ST['m'] = m
        ST['h'] = h
        ST['n'] = n
        ST['inp'] = 0.

    return nb.NeuType(name='HH_neuron', requires={"ST": ST}, steps=update, vector_based=True)


if __name__ == '__main__':
    nb.profile.set(backend='numba', device='cpu', dt=0.02,
                   numerical_method='exponential', merge_ing=True)

    HH = define_hh(noise=1.)

    neu = nb.NeuGroup(HH, geometry=(1,), monitors=['sp', 'V', 'm', 'h', 'n'])
    net = nb.Network(neu)
    net.run(duration=100., inputs=[neu, 'ST.inp', 10.], report=True)

    ts = net.ts
    fig, gs = nb.visualize.get_figure(2, 1, 3, 12)

    fig.add_subplot(gs[0, 0])
    plt.plot(ts, neu.mon.V[:, 0], label='N')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, net.t_start + 0.1)
    plt.legend()

    fig.add_subplot(gs[1, 0])
    plt.plot(ts, neu.mon.m[:, 0], label='m')
    plt.plot(ts, neu.mon.h[:, 0], label='h')
    plt.plot(ts, neu.mon.n[:, 0], label='n')
    plt.legend()
    plt.xlim(-0.1, net.t_start + 0.1)
    plt.xlabel('Time (ms)')

    plt.show()
