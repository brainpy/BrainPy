# -*- coding: utf-8 -*-

import time

import jax
import jax.numpy as np
import jax.ops

import brainpy as bp

dt = 0.01


def HH_model(num=20000, jit=True):
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
    neu_state = np.zeros((6, num))
    jax.ops.index_update(neu_state, jax.ops.index[0], Vr)

    def neu_update_state(st):
        V = st[0]
        m = st[1]
        h = st[2]
        n = st[3]
        Isyn = st[4]

        alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
        beta = 4.0 * np.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        m = np.clip(m + dmdt * dt, 0., 1.)

        alpha = 0.07 * np.exp(-(V + 65) / 20.)
        beta = 1 / (1 + np.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        h = np.clip(h + dhdt * dt, 0., 1.)

        alpha = 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
        beta = 0.125 * np.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        n = np.clip(n + dndt * dt, 0., 1.)

        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        Icur = - INa - IK - IL
        dvdt = (Icur + Isyn) / C
        V += dvdt * dt

        sp = np.logical_and(V >= Vth, st[0] < Vth)

        st = jax.ops.index_update(st, jax.ops.index[0], V)
        st = jax.ops.index_update(st, jax.ops.index[1], m)
        st = jax.ops.index_update(st, jax.ops.index[2], h)
        st = jax.ops.index_update(st, jax.ops.index[3], n)
        st = jax.ops.index_update(st, jax.ops.index[4], 0.)
        st = jax.ops.index_update(st, jax.ops.index[5], sp)
        return st

    if jit:
        neu_update_state = jax.jit(neu_update_state)

    sparseness = 0.05

    conn = bp.connect.FixedProb(sparseness)
    conn.set_size(num, num)
    conn(np.arange(num), np.arange(num))
    # conn.set_requires(['post2syn', 'pre2syn', 'post_slice_syn'])
    conn.set_requires(['post2syn', 'pre2syn', ])
    pre_ids = np.array(conn.pre_ids)
    pre2syn = [jax.ops.index[ids] for ids in conn.pre2syn]
    post2syn = [np.array(ids) for ids in conn.post2syn]

    # 's': 0., 'last_sp': -1e3
    syn_state = np.zeros((2, len(pre_ids)))

    g_max = 0.42
    E = 0.
    alpha = 0.98
    beta = 0.18
    T = 0.5
    T_duration = 0.5

    def syn_update_state(syn_st, post_st, t):
        s_var = syn_st[0]
        last_sp = syn_st[1]
        pre_sp = post_st[5]

        def body_func(i, sp):
            for i in range(num):
                return jax.lax.cond(pre_sp[i] > 0.,
                                    lambda sp_arr: jax.ops.index_update(sp_arr, pre2syn[i], t),
                                    lambda sp_arr: sp_arr,
                                    sp)
        last_sp = jax.lax.fori_loop(0, num, body_func, last_sp)

        # s = st
        TT = ((t - last_sp) < T_duration) * T
        dsdt = alpha * TT * (1 - s_var) - beta * s_var
        s_var += dsdt * dt
        syn_st = jax.ops.index_update(syn_st, jax.ops.index[0], s_var)

        g = np.zeros(num)
        for i in range(num):
            s = s_var[post2syn[i]]
            g = jax.ops.index_update(g, jax.ops.index[i], np.sum(s))
        post_val = - g_max * g * (post_st[0] - E)
        post_st = jax.ops.index_add(post_st, jax.ops.index[4], post_val)

        return syn_st, post_st

    if jit:
        syn_update_state = jax.jit(syn_update_state,)

    duration = 10.

    t0 = time.time()

    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    for ti in range(tlen):
        t = ts[ti]
        syn_state, neu_state = syn_update_state(syn_state, neu_state, t)
        neu_state = neu_update_state(neu_state)
        if (ti + 1) * 10 % tlen == 0:
            t1 = time.time()
            print('{} percent {} s'.format((ti + 1) / tlen * 100, t1 - t0))


HH_model(1000, jit=True)

