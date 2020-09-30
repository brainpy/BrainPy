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
nb.set_num_threads(8)


# from numba import nb.prange
dt = 0.01
npbrain.profile.set_dt(dt)


def HH_model():

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
    num = 2000
    delay = 10
    ra = True

    pre_ids, post_ids, anchors = nn.connect.fixed_prob(num, num, sparseness, False)

    neu_state = nn.init_neu_state(num, ['V', 'm', 'h', 'n'], recarray=ra)
    neu_state.V = Vr

    @nn.integrate(method=method, signature='f8[:](f8[:], f8, f8[:])')
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        return alpha * (1 - m) - beta * m

    @nn.integrate(method=method, signature='f8[:](f8[:], f8, f8[:])')
    def int_h(h, t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        return alpha * (1 - h) - beta * h

    @nn.integrate(method=method, signature='f8[:](f8[:], f8, f8[:])')
    def int_n(n, t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        return alpha * (1 - n) - beta * n

    @nn.integrate(method=method, signature='f8[:](f8[:], f8, f8[:], f8[:])')
    def int_V(V, t, Icur, Isyn):
        return (Icur + Isyn) / C

    @nn.helper.autojit
    def neu_update_state(st, t):
        m = nn.clip(int_m(st.m, t, st.V), 0., 1.)
        h = nn.clip(int_h(st.h, t, st.V), 0., 1.)
        n = nn.clip(int_n(st.n, t, st.V), 0., 1.)
        INa = g_Na * m ** 3 * h * (st.V - E_Na)
        IK = g_K * n ** 4 * (st.V - E_K)
        IL = g_Leak * (st.V - E_Leak)
        Icur = - INa - IK - IL
        V = int_V(st.V, t, Icur, st.syn_val)
        st.V = V
        st.m = m
        st.h = h
        st.n = n
        st.syn_val = 0.
        above_threshold = (st.V >= Vth).astype(np.float64)
        spike_st = above_threshold * (1. - st.above_th)
        spike_idx = np.where(spike_st > 0.)[0]
        st.above_th = above_threshold
        st.spike = spike_st
        st.spike_time[spike_idx] = t

    syn_state = nn.init_syn_state(len(pre_ids),
                                  variables={'s': 0., 'last_sp': -1e3},
                                  recarray=ra)
    post_state = nn.init_delay_state(num, delay)

    g_max = 0.42
    E = 0.
    alpha = 0.98
    beta = 0.18
    T = 0.5
    T_duration = 0.5

    @nn.integrate(signature='f8[:](f8[:], f8, f8[:])')
    def int_s(s, t, TT):
        return alpha * TT * (1 - s) - beta * s

    @nn.helper.autojit
    def syn_update_state(syn_st, post_st, t, pre_state):
        # calculate synaptic state
        spike_idx = np.where(pre_state.spike > 0.)[0]
        for i in spike_idx:
            idx = anchors[:, i]
            syn_st.last_sp[idx[0]: idx[1]] = t
        TT = ((t - syn_st.last_sp) < T_duration) * T
        syn_st.s = nn.clip(int_s(syn_st.s, t, TT), 0., 1.)
        # get post-synaptic values
        g = np.zeros(num)
        for i in nb.prange(num):
            idx = anchors[:, i]
            post_idx = post_ids[idx[0]: idx[1]]
            g[post_idx] += syn_st.s[idx[0]: idx[1]]
        didx = post_st.didx
        post_st.g[didx[0]] = g

    @nn.helper.autojit
    def syn_output_synapse(post_st, post_state):
        didx = post_st.didx
        g_val = post_st.g[didx[1]]
        post_val = - g_max * g_val * (post_state.V - E)
        post_state.syn_val += post_val

    duration = 100.

    t0 = time.time()
    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    for ti in range(tlen):
        t = ts[ti]
        syn_update_state(syn_state, post_state, t, neu_state)
        syn_output_synapse(post_state, neu_state)
        post_state.didx[0] = (post_state.didx[0] + 1) / post_state.dlen
        post_state.didx[1] = (post_state.didx[1] + 1) / post_state.dlen
        neu_update_state(neu_state, t)
        if (ti + 1) * 20 % tlen == 0:
            t1 = time.time()
            print('{} percent {} s'.format((ti + 1) / tlen, t1 - t0))
            t0 = time.time()



if __name__ == '__main__':
    HH_model()


