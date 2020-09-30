# -*- coding: utf-8 -*-

import time
import numpy as np
# import npbrain as nn
import numba as nb
from numba.typed import List

dt = 0.02


# parallel = True
# if parallel:
#     npbrain.profile.set_backend('numba-pa')
# else:
#     npbrain.profile.set_backend('numba')
# nn.profile.set_backend('numpy')

# from numba import nb.prange
# dt = 0.01
# npbrain.profile.set_dt(dt)


def clip(a, a_min, a_max):
    a = np.maximum(a, a_min)
    a = np.minimum(a, a_max)
    return a


def define_hh_neuron(E_Na=50., g_Na=120., E_K=-77., g_K=36.,
                     E_Leak=-54.387, g_Leak=0.03, C=1.0, Vr=-65., Vth=20.):
    def neu_update_state(st, t):
        V = st[0]
        m = st[1]
        h = st[2]
        n = st[3]
        sp = st[4]
        sp_t = st[5]
        input = st[6]

        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        m = m + dmdt * dt

        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        h = h + dhdt * dt

        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        n = n + dndt * dt

        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + input) / C
        V += dvdt * dt

        st[0] = V
        st[1] = m
        st[2] = h
        st[3] = n
        if V > Vth:
            st[4] = 1.
            st[5] = t
        else:
            st[4] = 0.
        st[6] = 0.

    return neu_update_state


def define_ampa_synapse(g_max=0.10, E=0., tau_decay=2.0):
    def update(S, t, pre_sp):
        s = S[0]
        s = s - s / tau_decay * dt + pre_sp
        S[0] = s

    def output(S, t, post_input, post_V):
        s = S[0]
        post_input -= g_max * s * (post_V - E)

    return update, output


def define_cpu_version(E_Na=50., g_Na=120., E_K=-77., g_K=36., E_Leak=-54.387, g_Leak=0.03,
                       C=1.0, Vth=20., g_max=0.10, E=0., tau_decay=2.0):
    @nb.njit
    def neu_update(pre_st, i):
        V = pre_st[i, 0]
        m = pre_st[i, 1]
        h = pre_st[i, 2]
        n = pre_st[i, 3]
        sp = pre_st[i, 4]
        input = pre_st[i, 5]

        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        m = m + dmdt * dt

        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        h = h + dhdt * dt

        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        n = n + dndt * dt

        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + input) / C
        V += dvdt * dt

        pre_st[i, 0] = V
        pre_st[i, 1] = m
        pre_st[i, 2] = h
        pre_st[i, 3] = n
        if V > Vth:
            pre_st[i, 4] = sp
        else:
            pre_st[i, 4] = 0.
        pre_st[i, 5] = 10.

    @nb.njit
    def syn_update(syn_st, pre_st, delay_in, syn_i, pre_i):
        s = syn_st[delay_in, syn_i, 0]
        s = s - s / tau_decay * dt + pre_st[pre_i, 4]
        syn_st[delay_in, syn_i, 0] = s

    @nb.njit
    def syn_output(syn_st, post_st, delay_out, syn_i, post_i):
        post_st[post_i, 5] -= g_max * syn_st[delay_out, syn_i, 0] * (post_st[post_i, 0] - E)

    return neu_update, syn_update, syn_output


def define_parallel_version(E_Na=50., g_Na=120., E_K=-77., g_K=36., E_Leak=-54.387, g_Leak=0.03,
                            C=1.0, Vth=20., g_max=0.10, E=0., tau_decay=2.0):
    @nb.njit(parallel=True)
    def neu_update(pre_st, i):
        V = pre_st[i, 0]
        m = pre_st[i, 1]
        h = pre_st[i, 2]
        n = pre_st[i, 3]
        sp = pre_st[i, 4]
        input = pre_st[i, 5]

        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        m = m + dmdt * dt

        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        h = h + dhdt * dt

        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        n = n + dndt * dt

        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + input) / C
        V += dvdt * dt

        pre_st[i, 0] = V
        pre_st[i, 1] = m
        pre_st[i, 2] = h
        pre_st[i, 3] = n
        if V > Vth:
            pre_st[i, 4] = sp
        else:
            pre_st[i, 4] = 0.
        pre_st[i, 5] = 10.

    @nb.njit(parallel=True)
    def syn_update(syn_st, pre_st, delay_in, syn_i, pre_i):
        s = syn_st[delay_in, syn_i, 0]
        s = s - s / tau_decay * dt + pre_st[pre_i, 4]
        syn_st[delay_in, syn_i, 0] = s

    @nb.njit(parallel=True)
    def syn_output(syn_st, post_st, delay_out, syn_i, post_i):
        post_st[post_i, 5] -= g_max * syn_st[delay_out, syn_i, 0] * (post_st[post_i, 0] - E)

    return neu_update, syn_update, syn_output


def define_cuda_version(E_Na=50., g_Na=120., E_K=-77., g_K=36., E_Leak=-54.387, g_Leak=0.03,
                        C=1.0, Vth=20., g_max=0.10, E=0., tau_decay=2.0):
    @nb.njit(parallel=True)
    def neu_update(pre_st, i):
        V = pre_st[i, 0]
        m = pre_st[i, 1]
        h = pre_st[i, 2]
        n = pre_st[i, 3]
        sp = pre_st[i, 4]
        input = pre_st[i, 5]

        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        m = m + dmdt * dt

        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        h = h + dhdt * dt

        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        n = n + dndt * dt

        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + input) / C
        V += dvdt * dt

        pre_st[i, 0] = V
        pre_st[i, 1] = m
        pre_st[i, 2] = h
        pre_st[i, 3] = n
        if V > Vth:
            pre_st[i, 4] = sp
        else:
            pre_st[i, 4] = 0.
        pre_st[i, 5] = 10.

    @nb.njit(parallel=True)
    def syn_update(syn_st, pre_st, delay_in, syn_i, pre_i):
        s = syn_st[delay_in, syn_i, 0]
        s = s - s / tau_decay * dt + pre_st[pre_i, 4]
        syn_st[delay_in, syn_i, 0] = s

    @nb.njit(parallel=True)
    def syn_output(syn_st, post_st, delay_out, syn_i, post_i):
        post_st[post_i, 5] -= g_max * syn_st[delay_out, syn_i, 0] * (post_st[post_i, 0] - E)

    return neu_update, syn_update, syn_output


def cpu_version():
    g_max = 0.10
    E = 0.
    tau_decay = 2.0

    dt = 0.02
    num_pre = 1000
    num_post = 1000
    pre_state = np.zeros((num_pre, 6))
    post_state = np.zeros((num_post, 6))

    delay = 10
    delay_in = 0
    delay_out = 9

    pre2syn = nb.typed.List()
    for i in range(num_pre):
        pre2syn.append(nb.typed.List.empty_list(nb.types.int64))
    post2syn = nb.typed.List()
    for i in range(num_post):
        post2syn.append(nb.typed.List.empty_list(nb.types.int64))
    syn_idx = np.int64(0)
    rand = np.random.random((num_pre, num_post))
    for i in range(num_pre):
        for j in range(num_post):
            if rand[i, j] < 0.1:
                pre2syn[i].append(syn_idx)
                post2syn[j].append(syn_idx)
                syn_idx += 1
    syn_state = np.zeros((delay, syn_idx, 1))

    @nb.njit(parallel=True)
    def main(pre_st, post_st, syn_st, pre2syn, post2syn):

        # pre
        for i in nb.prange(num_pre):
            V = pre_st[i, 0]
            m = pre_st[i, 1]
            h = pre_st[i, 2]
            n = pre_st[i, 3]
            sp = pre_st[i, 4]
            input = pre_st[i, 5]

            alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
            beta = 4.0 * np.exp(-(V + 65) / 18)
            dmdt = alpha * (1 - m) - beta * m
            m = m + dmdt * dt

            alpha = 0.07 * np.exp(-(V + 65) / 20.)
            beta = 1 / (1 + np.exp(-(V + 35) / 10))
            dhdt = alpha * (1 - h) - beta * h
            h = h + dhdt * dt

            alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
            beta = 0.125 * np.exp(-(V + 65) / 80)
            dndt = alpha * (1 - n) - beta * n
            n = n + dndt * dt

            INa = g_Na * m ** 3 * h * (V - E_Na)
            IK = g_K * n ** 4 * (V - E_K)
            IL = g_Leak * (V - E_Leak)
            dvdt = (- INa - IK - IL + input) / C
            V += dvdt * dt

            pre_st[i, 0] = V
            pre_st[i, 1] = m
            pre_st[i, 2] = h
            pre_st[i, 3] = n
            if V > Vth:
                pre_st[i, 4] = sp
            else:
                pre_st[i, 4] = 0.
            pre_st[i, 5] = 10.

        # synapse
        for i in nb.prange(num_pre):
            for idx in pre2syn[i]:
                s = syn_st[delay_in, idx, 0]
                s = s - s / tau_decay * dt + pre_st[i, 4]
                syn_st[delay_in, idx, 0] = s
        for i in nb.prange(num_post):
            for idx in post2syn[i]:
                post_st[i, 5] -= g_max * syn_st[delay_out, idx, 0] * (post_st[i, 0] - E)

        # post
        for i in nb.prange(num_post):
            V = post_st[i, 0]
            m = post_st[i, 1]
            h = post_st[i, 2]
            n = post_st[i, 3]
            sp = post_st[i, 4]
            input = post_st[i, 5]

            alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
            beta = 4.0 * np.exp(-(V + 65) / 18)
            dmdt = alpha * (1 - m) - beta * m
            m = m + dmdt * dt

            alpha = 0.07 * np.exp(-(V + 65) / 20.)
            beta = 1 / (1 + np.exp(-(V + 35) / 10))
            dhdt = alpha * (1 - h) - beta * h
            h = h + dhdt * dt

            alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
            beta = 0.125 * np.exp(-(V + 65) / 80)
            dndt = alpha * (1 - n) - beta * n
            n = n + dndt * dt

            INa = g_Na * m ** 3 * h * (V - E_Na)
            IK = g_K * n ** 4 * (V - E_K)
            IL = g_Leak * (V - E_Leak)
            dvdt = (- INa - IK - IL + input) / C
            V += dvdt * dt

            post_st[i, 0] = V
            post_st[i, 1] = m
            post_st[i, 2] = h
            post_st[i, 3] = n
            if V > Vth:
                post_st[i, 4] = sp
            else:
                post_st[i, 4] = 0.
            post_st[i, 5] = 0.

    duration = 1000.

    t0 = time.time()
    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    for ti in range(tlen):
        main(pre_state, post_state, syn_state, pre2syn, post2syn)
        delay_in = (delay_in + 1) % delay
        delay_out = (delay_out + 1) % delay
        if (ti + 1) * 20 % tlen == 0:
            t1 = time.time()
            print('{} percent {:.4f} s'.format((ti + 1) / tlen, t1 - t0))


def cpu_version2():
    num_pre = 1000
    num_post = 1000
    pre_state = np.zeros((num_pre, 6))
    pre_state[:, 0] = -65.
    post_state = np.zeros((num_post, 6))

    delay = 10
    delay_in = 0
    delay_out = 9

    pre2syn = nb.typed.List()
    for i in range(num_pre):
        pre2syn.append(nb.typed.List.empty_list(nb.types.int64))
    post2syn = nb.typed.List()
    for i in range(num_post):
        post2syn.append(nb.typed.List.empty_list(nb.types.int64))
    syn_idx = np.int64(0)
    rand = np.random.random((num_pre, num_post))
    for i in range(num_pre):
        for j in range(num_post):
            if rand[i, j] < 0.1:
                pre2syn[i].append(syn_idx)
                post2syn[j].append(syn_idx)
                syn_idx += 1
    syn_state = np.zeros((delay, syn_idx, 1))

    # @nb.njit
    @nb.njit(parallel=True)
    def main(pre_st, post_st, syn_st, pre2syn, post2syn, delay_in, delay_out, t):
        # pre
        for i in nb.prange(num_pre):
            neu_update(pre_st, i)
        # synapse
        for i in nb.prange(num_pre):
            for idx in pre2syn[i]:
                syn_update(syn_st, pre_st, delay_in, idx, i)
        for i in nb.prange(num_post):
            for idx in post2syn[i]:
                syn_output(syn_st, post_st, delay_out, idx, i)
        # post
        for i in nb.prange(num_post):
            neu_update(post_st, i)

    # neu_update, syn_update, syn_output = define_cpu_version()
    neu_update, syn_update, syn_output = define_parallel_version()

    duration = 1000.
    t0 = time.time()
    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    for ti in range(tlen):
        main(pre_state, post_state, syn_state, pre2syn, post2syn, delay_in, delay_out, ts[ti])
        delay_in = (delay_in + 1) % delay
        delay_out = (delay_out + 1) % delay
        if (ti + 1) * 20 % tlen == 0:
            t1 = time.time()
            print('{} percent {:.4f} s'.format((ti + 1) / tlen, t1 - t0))


# cpu_version()
cpu_version2()
