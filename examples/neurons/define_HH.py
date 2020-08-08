# -*- coding: utf-8 -*-

import numpy as np
import npbrain as nn


def HH(geometry, method=None, noise=0., E_Na=50., g_Na=120., E_K=-77.,
       g_K=36., E_Leak=-54.387, g_Leak=0.03, C=1.0, Vr=-65., Vth=20.):

    var2index = {'V': 0, 'm': 1, 'h': 2, 'n': 3}
    num, geometry = nn.format_geometry(geometry)
    state = nn.init_neu_state(4, num)

    @nn.integrate(method=method)
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m

    @nn.integrate(method=method)
    def int_h(h, t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        return alpha * (1 - h) - beta * h

    @nn.integrate(method=method)
    def int_n(n, t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        return alpha * (1 - n) - beta * n

    @nn.integrate(method=method, noise=noise / C)
    def int_V(V, t, Icur, Isyn):
        return (Icur + Isyn) / C

    def update_state(neu_state, t):
        V, Isyn = neu_state[0], neu_state[-1]
        m = nn.clip(int_m(neu_state[1], t, V), 0., 1.)
        h = nn.clip(int_h(neu_state[2], t, V), 0., 1.)
        n = nn.clip(int_n(neu_state[3], t, V), 0., 1.)
        INa = g_Na * m * m * m * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        Icur = - INa - IK - IL
        V = int_V(V, t, Icur, Isyn)
        neu_state[0] = V
        neu_state[1] = m
        neu_state[2] = h
        neu_state[3] = n
        nn.judge_spike(neu_state, Vth, t)

    return nn.Neurons(**locals())
