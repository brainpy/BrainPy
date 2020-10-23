# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

import brainpy as nb
import brainpy.numpy as np

nb.profile.set(backend='numba', device='cpu', dt=0.02, merge_ing=True,
               numerical_method='euler')

noise = 0.  # noise term
C = 1.0  # Membrane capacity per unit area (assumed constant).
g_Na = 120.  # Voltage-controlled conductance per unit area associated with the Sodium (Na) ion-channel.
E_Na = 50.  # The equilibrium potentials for the sodium ions.
E_K = -77.  # The equilibrium potentials for the potassium ions.
g_K = 36.  # Voltage-controlled conductance per unit area associated with the Potassium (K) ion-channel.
E_Leak = -54.387  # The equilibrium potentials for the potassium ions.
g_Leak = 0.03  # Conductance per unit area associated with the leak channels.
Vth = 20.  # membrane potential threshold for spike

ST = nb.types.NeuState(
    {'V': -65.,  # denotes membrane potential.
     'm': 0.,  # denotes potassium channel activation probability.
     'h': 0.,  # denotes sodium channel activation probability.
     'n': 0.,  # denotes sodium channel inactivation probability.
     'sp': 0.,  # denotes spiking state.
     'inp': 0.  # denotes synaptic input.
     }
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


HH = nb.NeuType(name='HH_neuron', requires={"ST": ST}, steps=update, vector_based=True)

if __name__ == '__main__':
    mon = HH.run(duration=100., monitors=['sp', 'V', 'm', 'h', 'n'], inputs=['ST.inp', 10.], report=True)
    ts = mon.ts
    fig, gs = nb.visualize.get_figure(2, 1, 3, 12)

    fig.add_subplot(gs[0, 0])
    plt.plot(ts, mon.V, label='N')
    plt.ylabel('Membrane potential')
    plt.xlim(-0.1, 100 + 0.1)
    plt.legend()

    fig.add_subplot(gs[1, 0])
    plt.plot(ts, mon.m, label='m')
    plt.plot(ts, mon.h, label='h')
    plt.plot(ts, mon.n, label='n')
    plt.legend()
    plt.xlim(-0.1, 100 + 0.1)
    plt.xlabel('Time (ms)')

    plt.show()
