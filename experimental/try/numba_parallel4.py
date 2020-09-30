# -*- coding: utf-8 -*-

import time
import numpy as np
import npbrain as nn
import numba as nb
npbrain.profile.define_signature = False

parallel = False
if parallel:
    npbrain.profile.set_backend('numba-pa')
else:
    npbrain.profile.set_backend('numba')
    # nn.profile.set_backend('numpy')


# from numba import nb.prange
dt = 0.01
npbrain.profile.set_dt(dt)


def HH_model(num=20000):

    E_Na = 50.
    g_Na = 120.
    E_K = -77.
    g_K = 36.
    E_Leak = -54.387
    g_Leak = 0.03
    C = 1.0
    Vr = -65.
    Vth = 20.

    method = 'euler'
    sparseness = 0.05
    delay = 10

    pre_ids, post_ids, anchors = nn.connect.fixed_prob(num, num, sparseness, False)

    neu_state = nn.init_neu_state(num, ['V', 'm', 'h', 'n'])
    neu_state.V[:] = Vr

    @nn.integrate(method=method, signature='f[:](f[:], f, f[:])')
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m

    @nn.integrate(method=method, signature='f[:](f[:], f, f[:])')
    def int_h(h, t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        return alpha * (1 - h) - beta * h

    @nn.integrate(method=method, signature='f[:](f[:], f, f[:])')
    def int_n(n, t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        return alpha * (1 - n) - beta * n

    @nn.integrate(method=method, signature='f[:](f[:], f, f[:], f[:])')
    def int_V(V, t, Icur, Isyn):
        return (Icur + Isyn) / C

    @nn.autojit
    def neu_update_state(st, t):
        m = nn.clip(int_m(st.m, t, st.V), 0., 1.)
        h = nn.clip(int_h(st.h, t, st.V), 0., 1.)
        n = nn.clip(int_n(st.n, t, st.V), 0., 1.)
        INa = g_Na * m ** 3 * h * (st.V - E_Na)
        IK = g_K * n ** 4 * (st.V - E_K)
        IL = g_Leak * (st.V - E_Leak)
        Icur = - INa - IK - IL
        V = int_V(st.V, t, Icur, st.syn_val)
        st.V[:] = V
        st.m[:] = m
        st.h[:] = h
        st.n[:] = n
        st.syn_val[:] = 0.
        above_threshold = (st.V >= Vth).astype(np.float64)
        spike_st = above_threshold * (1. - st.above_th)
        spike_idx = np.where(spike_st > 0.)[0]
        st.above_th[:] = above_threshold
        st.spike[:] = spike_st
        st.spike_time[spike_idx] = t


    syn_state = nn.init_syn_state(len(pre_ids), variables={'s': 0., 'last_sp': -1e3})
    delay_state = nn.init_delay_state(num, delay)

    g_max = 0.42
    E = 0.
    alpha = 0.98
    beta = 0.18
    T = 0.5
    T_duration = 0.5

    @nn.integrate(signature='f8[:](f8[:], f8, f8[:])')
    def int_s(s, t, TT):
        return alpha * TT * (1 - s) - beta * s

    @nn.autojit
    def syn_update_state(syn_st, delay_st, t, pre_st):
        # calculate synaptic state
        spike_idx = np.where(pre_st.spike > 0.)[0]
        for i in spike_idx:
            idx = anchors[:, i]
            syn_st.last_sp[idx[0]: idx[1]] = t
        TT = ((t - syn_st.last_sp) < T_duration).astype(np.float64) * T
        # s = nn.clip(int_s(syn_st.s, t, TT), 0., 1.)
        s = int_s(syn_st.s, t, TT)
        syn_st.s[:] = s
        # get post-synaptic values
        g = np.zeros(num)
        for i in nb.prange(num):
            idx = anchors[:, i]
            post_idx = post_ids[idx[0]: idx[1]]
            g[post_idx] += syn_st.s[idx[0]: idx[1]]
        didx = delay_st.didx
        delay_st.g[didx[0]] = g

    @nn.autojit
    def syn_output_synapse(delay_st, post_st):
        didx = delay_st.didx
        g_val = delay_st.g[didx[1]]
        post_val = - g_max * g_val * (post_st.V - E)
        post_st.syn_val[:] += post_val


    duration = 10.
    # duration = 0.1

    t0 = time.time()

    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    for ti in range(tlen):
        t = ts[ti]
        syn_update_state(syn_state, delay_state, t, neu_state)
        syn_output_synapse(delay_state, neu_state)
        delay_state.didx[0] = (delay_state.didx[0] + 1) / delay_state.dlen
        delay_state.didx[1] = (delay_state.didx[1] + 1) / delay_state.dlen
        neu_update_state(neu_state, t)
        if (ti + 1) * 20 % tlen == 0:
            t1 = time.time()
            print('{} percent {} s'.format((ti + 1) / tlen, t1 - t0))
            t0 = time.time()

    # int_V.inspect_types()


if __name__ == '__main__':
    HH_model(5000)


