# -*- coding: utf-8 -*-

import time
from collections import namedtuple

import numpy as np
from numba import njit

dt = 0.01


def namedtupled_for_parameters(num=20000):
    global MyRec
    MyRec = namedtuple("MyRec", ['E_Na', 'g_Na', 'E_K', 'g_K', 'E_Leak',
                                 'g_Leak', 'C', 'Vr', 'Vth'])

    @njit
    def make():
        return MyRec(E_Na=50., g_Na=120., E_K=-77., g_K=36., E_Leak=-54.387,
                     g_Leak=0.03, C=1.0, Vr=-65., Vth=20.)

    pars = make()

    neu_state = np.zeros((4 + 5, num))
    neu_state[0] = pars.Vr
    neu_state[-2] = -1e5
    neu_state[-5] = 1.

    @njit
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        return m + dmdt * dt

    @njit
    def int_h(h, t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        return h + dhdt * dt

    @njit
    def int_n(n, t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        return n + dndt * dt

    @njit
    def judge_spike(neu_state, vth, t):
        above_threshold = (neu_state[0] >= vth).astype(np.float64)
        prev_above_th = neu_state[-4]
        spike_st = above_threshold * (1. - prev_above_th)
        spike_idx = np.where(spike_st > 0.)[0]
        neu_state[-4] = above_threshold
        neu_state[-3] = spike_st
        neu_state[-2][spike_idx] = t
        return spike_idx

    @njit
    def neu_update_state(st, t, pars):
        V = st[0]
        m = st[1]
        h = st[2]
        n = st[3]

        m = int_m(m, t, V)
        h = int_h(h, t, V)
        n = int_n(n, t, V)

        INa = pars.g_Na * m ** 3 * h * (V - pars.E_Na)
        IK = pars.g_K * n ** 4 * (V - pars.E_K)
        IL = pars.g_Leak * (V - pars.E_Leak)
        dvdt = (- INa - IK - IL + st[-1]) / pars.C
        V += dvdt * dt

        st[0] = V
        st[1] = m
        st[2] = h
        st[3] = n
        st[-1] = 0.
        judge_spike(st, pars.Vth, t)

    duration = 100.
    t0 = time.time()
    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    for ti in range(tlen):
        t = ts[ti]
        neu_update_state(neu_state, t, pars)
        if (ti + 1) * 20 % tlen == 0:
            t1 = time.time()
            print('{} percent {} s'.format((ti + 1) / tlen, t1 - t0))


def namedtupled_for_neuron_state(num=20000):
    global nt2
    nt2 = namedtuple("nt2", ['V', 'm', 'h', 'n', 'not_ref', 'above_th',
                             'spike', 'spike_time', 'Isyn'])

    @njit
    def make(num):
        return nt2(V=np.ones((num,)) * Vr,
                   m=np.zeros((num,)),
                   h=np.zeros((num,)),
                   n=np.zeros((num,)),
                   not_ref=np.ones((num,)),
                   above_th=np.zeros((num,)),
                   spike=np.zeros((num,)),
                   spike_time=np.ones((num,)) * -1e5,
                   Isyn=np.zeros((num,)))

    E_Na = 50.
    g_Na = 120.
    E_K = -77.
    g_K = 36.
    E_Leak = -54.387
    g_Leak = 0.03
    C = 1.0
    Vr = -65.
    Vth = 20.


    neu_state = make(num)

    @njit
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        return m + dmdt * dt

    @njit
    def int_h(h, t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        return h + dhdt * dt

    @njit
    def int_n(n, t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        return n + dndt * dt

    @njit
    def judge_spike(neu_state, vth, t):
        above_threshold = (neu_state.V >= vth).astype(np.float64)
        prev_above_th = neu_state.above_th
        spike_st = above_threshold * (1. - prev_above_th)
        spike_idx = np.where(spike_st > 0.)[0]
        neu_state.above_th[:] = above_threshold
        neu_state.spike[:] = spike_st
        neu_state.spike_time[spike_idx] = t
        return spike_idx

    @njit
    def neu_update_state(st, t):
        V = st.V
        m = st.m
        h = st.h
        n = st.n

        m = int_m(m, t, V)
        h = int_h(h, t, V)
        n = int_n(n, t, V)

        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + st[-1]) / C
        V += dvdt * dt

        st.V[:] = V
        st.m[:] = m
        st.h[:] = h
        st.n[:] = n
        st.Isyn[:] = 0.
        judge_spike(st, Vth, t)

    duration = 100.
    t0 = time.time()
    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    for ti in range(tlen):
        t = ts[ti]
        neu_update_state(neu_state, t)
        if (ti + 1) * 20 % tlen == 0:
            t1 = time.time()
            print('{} percent {} s'.format((ti + 1) / tlen, t1 - t0))


if __name__ == '__main__':
    namedtupled_for_parameters()
    # namedtupled_for_neuron_state()
