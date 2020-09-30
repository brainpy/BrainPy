# -*- coding: utf-8 -*-

import time
import numpy as np
import npbrain as nn
import numba as nb

npbrain.profile.define_signature = False

parallel = True
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

    neu_state = nn.init_neu_state(num, len(['V', 'm', 'h', 'n']))
    neu_state[0] = Vr

    @nn.autojit
    def neu_update_state(st, t):
        V = st[0]
        m = st[1]
        h = st[2]
        n = st[3]

        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        m = nn.clip(m + dmdt * dt, 0., 1.)

        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        h = nn.clip(h + dhdt * dt, 0., 1.)

        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        n = nn.clip(n + dndt * dt, 0., 1.)

        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + st[-1]) / C
        V += dvdt * dt

        st[0] = V
        st[1] = m
        st[2] = h
        st[3] = n
        above_threshold = (V >= Vth).astype(np.float64)
        spike_st = above_threshold * (1. - st[-4])
        spike_idx = np.where(spike_st > 0.)[0]
        st[-4] = above_threshold
        st[-3] = spike_st
        st[-2][spike_idx] = t

    duration = 10.

    t0 = time.time()
    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    for ti in range(tlen):
        t = ts[ti]
        neu_update_state(neu_state, t)
        if (ti + 1) * 20 % tlen == 0:
            t1 = time.time()
            print('{} percent {:.4f} s'.format((ti + 1) / tlen, t1 - t0))


def HH_model_1_1(num=20000):
    E_Na = 50.
    g_Na = 120.
    E_K = -77.
    g_K = 36.
    E_Leak = -54.387
    g_Leak = 0.03
    C = 1.0
    Vr = -65.
    Vth = 20.

    neu_state = nn.init_neu_state(num, len(['V', 'm', 'h', 'n']))
    neu_state[0] = Vr

    @nn.autojit
    def neu_update_state(st, t):
        V = st[0]
        m = st[1]
        h = st[2]
        n = st[3]

        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        m = nn.clip(m + dmdt * dt, 0., 1.)

        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        h = nn.clip(h + dhdt * dt, 0., 1.)

        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        n = nn.clip(n + dndt * dt, 0., 1.)

        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + st[-1]) / C
        V += dvdt * dt

        st[0] = V
        st[1] = m
        st[2] = h
        st[3] = n
        above_threshold = (V >= Vth).astype(np.float64)
        spike_st = above_threshold * (1. - st[-4])
        spike_idx = np.where(spike_st > 0.)[0]
        st[-4] = above_threshold
        st[-3] = spike_st
        st[-2][spike_idx] = t

    @nn.autojit
    def run(neust, ts):
        tlen = len(ts)
        for ti in nb.prange(tlen):
            t = ts[ti]
            neu_update_state(neust, t)

    duration = 10.

    t0 = time.time()
    ts = np.arange(0, duration, dt)
    run(neu_state, ts)
    t1 = time.time()
    print('used {:.4f} s'.format(t1 - t0))


def HH_model2(num=20000):
    E_Na = 50.
    g_Na = 120.
    E_K = -77.
    g_K = 36.
    E_Leak = -54.387
    g_Leak = 0.03
    C = 1.0
    Vr = -65.
    Vth = 20.

    neu_state = nn.init_neu_state(num, len(['V', 'm', 'h', 'n']))
    neu_state[0] = Vr

    @nn.autojit
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        return m + dmdt * dt

    @nn.autojit
    def int_h(h, t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        return h + dhdt * dt

    @nn.autojit
    def int_n(n, t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        return n + dndt * dt

    @nn.autojit
    def neu_update_state(st, t):
        V = st[0]
        m = st[1]
        h = st[2]
        n = st[3]

        m = nn.clip(int_m(m, t, V), 0., 1.)
        h = nn.clip(int_h(h, t, V), 0., 1.)
        n = nn.clip(int_n(n, t, V), 0., 1.)

        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + st[-1]) / C
        V += dvdt * dt

        st[0] = V
        st[1] = m
        st[2] = h
        st[3] = n
        nn.judge_spike(st, Vth, t)

    duration = 10.

    t0 = time.time()
    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    for ti in range(tlen):
        t = ts[ti]
        neu_update_state(neu_state, t)
        if (ti + 1) * 20 % tlen == 0:
            t1 = time.time()
            print('{} percent {:.4f} s'.format((ti + 1) / tlen, t1 - t0))


def HH_model3(num=20000):
    E_Na = 50.
    g_Na = 120.
    E_K = -77.
    g_K = 36.
    E_Leak = -54.387
    g_Leak = 0.03
    C = 1.0
    Vr = -65.
    Vth = 20.

    neu_state = nn.init_neu_state(num, len(['V', 'm', 'h', 'n']))
    neu_state[0] = Vr

    @nn.integrate
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        return dmdt

    @nn.integrate
    def int_h(h, t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        return dhdt

    @nn.integrate
    def int_n(n, t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        return dndt

    @nn.autojit
    def neu_update_state(st, t):
        V = st[0]
        m = st[1]
        h = st[2]
        n = st[3]

        m = nn.clip(int_m(m, t, V), 0., 1.)
        h = nn.clip(int_h(h, t, V), 0., 1.)
        n = nn.clip(int_n(n, t, V), 0., 1.)

        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + st[-1]) / C
        V += dvdt * dt

        st[0] = V
        st[1] = m
        st[2] = h
        st[3] = n
        nn.judge_spike(st, Vth, t)

    duration = 10.

    t0 = time.time()
    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    for ti in range(tlen):
        t = ts[ti]
        neu_update_state(neu_state, t)
        if (ti + 1) * 20 % tlen == 0:
            t1 = time.time()
            print('{} percent {:.4f} s'.format((ti + 1) / tlen, t1 - t0))


def HH_model_4_1(num=20000):
    E_Na = 50.
    g_Na = 120.
    E_K = -77.
    g_K = 36.
    E_Leak = -54.387
    g_Leak = 0.03
    C = 1.0
    Vr = -65.
    Vth = 20.

    neu_state = nn.init_neu_state(num, len(['V', 'm', 'h', 'n']))
    neu_state[0] = Vr

    @nn.autojit
    def int_m(m, t, V):
        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        return m + dmdt * dt

    @nn.autojit
    def int_h(h, t, V):
        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        return h + dhdt * dt

    @nn.autojit
    def int_n(n, t, V):
        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        return n + dndt * dt

    @nn.autojit
    def neu_update_state(st, t):
        V = st[0]
        m = st[1]
        h = st[2]
        n = st[3]

        m = nn.clip(int_m(m, t, V), 0., 1.)
        h = nn.clip(int_h(h, t, V), 0., 1.)
        n = nn.clip(int_n(n, t, V), 0., 1.)

        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + st[-1]) / C
        V += dvdt * dt

        st[0] = V
        st[1] = m
        st[2] = h
        st[3] = n
        st[-1] = 0.
        nn.judge_spike(st, Vth, t)

    sparseness = 0.05
    delay = 10

    pre_ids, post_ids, anchors = nn.connect.fixed_prob(num, num, sparseness, False)

    syn_state = nn.init_syn_state(len(pre_ids), variables=[('s', 0.), ('last_sp', -1e3)])
    delay_state = nn.init_delay_state(num, delay)
    delay_len = nn.format_delay(delay, dt=dt)
    var2index = {'g_in': delay_len - 1, 'g_out': 0}

    g_max = 0.42
    E = 0.
    alpha = 0.98
    beta = 0.18
    T = 0.5
    T_duration = 0.5

    @nn.autojit
    def syn_update_state(syn_st, delay_st, delay_idx, t, pre_st):
        s = syn_st[0]
        last_sp = syn_st[1]
        pre_sp = pre_st[-3]
        # calculate synaptic state
        spike_idx = np.where(pre_sp > 0.)[0]
        for i in spike_idx:
            idx = anchors[:, i]
            last_sp[idx[0]: idx[1]] = t
        TT = ((t - last_sp) < T_duration).astype(np.float64) * T
        dsdt = alpha * TT * (1 - s) - beta * s
        # s = nn.clip(s + dsdt * dt, 0., 1.)
        s = s + dsdt * dt
        syn_st[0] = s
        syn_st[1] = last_sp
        # get post-synaptic values
        g = np.zeros(num)
        for i in nb.prange(num):
            idx = anchors[:, i]
            post_idx = post_ids[idx[0]: idx[1]]
            g[post_idx] += s[idx[0]: idx[1]]
        delay_st[delay_idx] = g

    @nn.autojit
    def syn_output_synapse(delay_st, out_idx, post_st):
        g_val = delay_st[out_idx]
        post_val = - g_max * g_val * (post_st[0] - E)
        post_st[-1] += post_val

    duration = 10.
    t0 = time.time()
    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    for ti in range(tlen):
        t = ts[ti]
        syn_update_state(syn_state, delay_state, var2index['g_in'], t, neu_state)
        syn_output_synapse(delay_state, var2index['g_out'], neu_state)
        var2index['g_in'] = (var2index['g_in'] + 1) % delay_len
        var2index['g_out'] = (var2index['g_out'] + 1) % delay_len
        neu_update_state(neu_state, t)
        if (ti + 1) * 20 % tlen == 0:
            t1 = time.time()
            print('{} percent {} s'.format((ti + 1) / tlen, t1 - t0))


if __name__ == '__main__':
    pass
    # HH_model(200000)  # 17.1030 s (without parallel), 9.0914 s (parallel)
    # HH_model_1_1(200000)  # 17.7037 s (without parallel), 12.2901 s (parallel)
    # HH_model2(200000)  # 9.2325 s (parallel)
    # HH_model3(200000)  # 11.7994 s (parallel)
    # HH_model_4_1(5000)  # 26.612202 s (without parallel), 21.31218 s (parallel)
    HH_model_4_1(20000)  # 346.67667 s (without parallel), 218.65979 s (parallel)
