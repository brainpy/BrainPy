# -*- coding: utf-8 -*-

import time

import numpy as onp
import jax
import jax.ops
import jax.numpy as np
from jax.experimental import loops

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

        jax.ops.index_update(st, jax.ops.index[0], V)
        jax.ops.index_update(st, jax.ops.index[1], m)
        jax.ops.index_update(st, jax.ops.index[2], h)
        jax.ops.index_update(st, jax.ops.index[3], n)
        jax.ops.index_update(st, jax.ops.index[4], 0.)
        jax.ops.index_update(st, jax.ops.index[5], sp)

    if jit:
        neu_update_state = jax.jit(neu_update_state)

    sparseness = 0.05

    conn = bp.connect.FixedProb(sparseness)
    conn.set_size(num, num)
    conn(np.arange(num), np.arange(num))
    conn.set_requires(['post_slice_syn', 'pre2syn'])
    # conn.make_post_slice_syn()
    # conn.make_pre2syn()
    pre_ids = np.array(conn.pre_ids)
    # pre2syn = [ids for ids in conn.pre2syn]
    pre2syn = conn.pre2syn
    pre2syn_slice = []
    s = 0
    for ids in enumerate(pre2syn):
        pre2syn_slice.append([s, s+ len(ids)])
        s += len(ids)
    pre2syn_slice = np.array(pre2syn_slice)
    pre2syn = np.array(onp.concatenate(pre2syn))
    post_slice_syn = np.array(conn.post_slice_syn)

    # 's': 0., 'last_sp': -1e3
    syn_state = np.zeros((2, len(pre_ids)))

    g_max = 0.42
    E = 0.
    alpha = 0.98
    beta = 0.18
    T = 0.5
    T_duration = 0.5

    def syn_update_state(syn_st, t, pre_st, post_st):
        st = syn_st[0]
        last_sp = syn_st[1]
        pre_sp = pre_st[5]

        # calculate synaptic state

        with loops.Scope() as s:
            s.pre_sp = pre_sp
            s.last_sp = last_sp
            s.sp = s.last_sp[0]
            s.pre2syn_slice = pre2syn_slice
            s.pre2syn = pre2syn
            s.slice = s.pre2syn_slice[0]
            s.idx = s.pre2syn[s.slice[0]: s.slice[1]]
            # s.idx = jax.lax.dynamic_slice(s.pre2syn, (s.slice[0],), (s.slice[1],))
            # s.idx = jax.lax.dynamic_slice(s.pre2syn, (s.slice[0], s.slice[1]),(2,))
            s.t = t

            for i in s.range(num):
                s.sp = s.pre_sp[i]
                for _ in s.cond_range(s.sp > 0.):
                    s.slice = s.pre2syn_slice[i]
                    s.idx = s.pre2syn[s.slice[0]: s.slice[1]]
                    jax.ops.index_update(s.last_sp, jax.ops.index[s.idx], s.t)

        last_sp = s.last_sp

        # s = st
        TT = ((t - last_sp) < T_duration) * T
        dsdt = alpha * TT * (1 - s) - beta * s
        s += dsdt * dt
        # syn_st[0] = s
        jax.ops.index_update(syn_st, jax.ops.index[0], s)

        # get post-synaptic values
        g = np.zeros(num)
        for i in range(num):
            idx = post_slice_syn[i]
            jax.ops.index_add(g, jax.ops.index[i], np.sum(s[idx[0]: idx[1]]))
        post_val = - g_max * g * (post_st[0] - E)
        jax.ops.index_add(post_st, jax.ops.index[4], post_val)

    if jit:
        syn_update_state = jax.jit(syn_update_state, static_argnums=(0, 1, 2, 3))

    duration = 10.

    t0 = time.time()

    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    for ti in range(tlen):
        t = ts[ti]
        syn_update_state(syn_state, t, neu_state, neu_state)
        neu_update_state(neu_state)
        if (ti + 1) * 10 % tlen == 0:
            t1 = time.time()
            print('{} percent {} s'.format((ti + 1) / tlen * 100, t1 - t0))
            t0 = time.time()


HH_model(100, jit=False)

