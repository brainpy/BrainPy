# -*- coding: utf-8 -*-

import time

import torch

import brainpy as bp

dt = 0.01


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

    # ['V', 'm', 'h', 'n', Isyn]
    neu_state = torch.zeros((6, num))
    neu_state[0] = Vr

    def neu_update_state(st):
        V = st[0]
        m = st[1]
        h = st[2]
        n = st[3]
        Isyn = st[4]

        alpha = 0.1 * (V + 40) / (1 - torch.exp(-(V + 40) / 10))
        beta = 4.0 * torch.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        m = torch.clip(m + dmdt * dt, 0., 1.)

        alpha = 0.07 * torch.exp(-(V + 65) / 20.)
        beta = 1 / (1 + torch.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        h = torch.clip(h + dhdt * dt, 0., 1.)

        alpha = 0.01 * (V + 55) / (1 - torch.exp(-(V + 55) / 10))
        beta = 0.125 * torch.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        n = torch.clip(n + dndt * dt, 0., 1.)

        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        Icur = - INa - IK - IL
        dvdt = (Icur + Isyn) / C
        V += dvdt * dt

        sp = torch.logical_and(V >= Vth, st[0] < Vth)

        st[0] = V
        st[1] = m
        st[2] = h
        st[3] = n
        st[4] = 0.
        st[5] = sp


    sparseness = 0.05

    conn = bp.connect.FixedProb(sparseness)
    conn.set_size(num, num)
    conn(torch.arange(num), torch.arange(num))
    conn.set_requires(['post_slice_syn', 'pre2syn'])
    pre_ids = torch.from_numpy(conn.pre_ids)
    # pre2syn = [ids for ids in conn.pre2syn]
    pre2syn = [torch.from_numpy(ids) for ids in conn.pre2syn]
    post_slice_syn = torch.from_numpy(conn.post_slice_syn)

    # 's': 0., 'last_sp': -1e3
    syn_state = torch.zeros((2, len(pre_ids)))

    g_max = 0.42
    E = 0.
    alpha = 0.98
    beta = 0.18
    T = 0.5
    T_duration = 0.5

    def syn_update_state(syn_st, t, pre_st, post_st):
        s = syn_st[0]
        last_sp = syn_st[1]
        pre_sp = pre_st[5]

        # calculate synaptic state
        for i in range(num):
            if pre_sp[i] > 0.:
                ids = pre2syn[i]
                last_sp[ids] = t

        # s = st
        TT = ((t - last_sp) < T_duration) * T
        dsdt = alpha * TT * (1 - s) - beta * s
        s += dsdt * dt
        syn_st[0] = s

        # get post-synaptic values
        g = torch.zeros(num)
        for i in range(num):
            idx = post_slice_syn[i]
            g[i] += torch.sum(s[idx[0]: idx[1]])
        post_val = - g_max * g * (post_st[0] - E)
        post_st[4] += post_val


    duration = 10.

    t0 = time.time()

    ts = torch.arange(0, duration, dt)
    tlen = len(ts)
    for ti in range(tlen):
        t = ts[ti]
        syn_update_state(syn_state, t, neu_state, neu_state)
        neu_update_state(neu_state)
        if (ti + 1) * 10 % tlen == 0:
            t1 = time.time()
            print('{} percent {} s'.format((ti + 1) / tlen * 100, t1 - t0))


HH_model(1000)

