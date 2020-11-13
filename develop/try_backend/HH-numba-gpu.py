# -*- coding: utf-8 -*-

import math
import time

from numba import cuda

import brainpy.numpy as np

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
    neu_state =  np.zeros((6, num))
    neu_state[0] = Vr
    neu_state[5] = 5.
    neu_state = cuda.to_device(neu_state)

    @cuda.jit
    def neu_update_state(st):
        tx = cuda.threadIdx.x
        ty = cuda.blockIdx.x
        bw = cuda.blockDim.x
        i = tx + ty * bw

        if i < st.shape[1]:
            V = st[0]
            m = st[1]
            h = st[2]
            n = st[3]
            Isyn = st[4]

            alpha = 0.1 * (V[i] + 40) / (1 - math.exp(-(V[i] + 40) / 10))
            beta = 4.0 * math.exp(-(V[i] + 65) / 18)
            dmdt = alpha * (1 - m[i]) - beta * m[i]
            m[i] = m[i] + dmdt * dt

            alpha = 0.07 * math.exp(-(V[i] + 65) / 20.)
            beta = 1 / (1 + math.exp(-(V[i] + 35) / 10))
            dhdt = alpha * (1 - h[i]) - beta * h[i]
            h[i] = h[i] + dhdt * dt

            alpha = 0.01 * (V[i] + 55) / (1 - math.exp(-(V[i] + 55) / 10))
            beta = 0.125 * math.exp(-(V[i] + 65) / 80)
            dndt = alpha * (1 - n[i]) - beta * n[i]
            n[i] = n[i] + dndt * dt

            INa = g_Na * m[i] ** 3 * h[i] * (V[i] - E_Na)
            IK = g_K * n[i] ** 4 * (V[i] - E_K)
            IL = g_Leak * (V[i] - E_Leak)
            Icur = - INa - IK - IL
            dvdt = (Icur + Isyn[i]) / C
            V[i] += dvdt * dt

    #
    # sparseness = 0.05
    #
    # conn = bp.connect.FixedProb(sparseness)
    # conn.set_size(num, num)
    # conn(np.arange(num), np.arange(num))
    # conn.set_requires(['post_slice_syn', 'pre2syn'])
    # pre_ids = conn.pre_ids
    # pre2syn = conn.pre2syn
    # pre2syn_slice = []
    # start = 0
    # for ids in conn.pre2syn:
    #     pre2syn_slice.append([start, start + len(ids)])
    #     start += len(ids)
    # pre2syn_slice = cuda.to_device(np.array(pre2syn_slice))
    # pre2syn = cuda.to_device(np.concatenate(pre2syn))
    # post_slice_syn = cuda.to_device(conn.post_slice_syn)
    #
    # # 's': 0., 'last_sp': -1e3
    # syn_state = np.zeros((2, len(pre_ids)))
    #
    # g_max = 0.42
    # E = 0.
    # alpha = 0.98
    # beta = 0.18
    # T = 0.5
    # T_duration = 0.5
    #
    # @cuda.jit
    # def syn_update_state(syn_st, t, pre_st, post_st):
    #     s = syn_st[0]
    #     last_sp = syn_st[1]
    #     pre_sp = pre_st[5]
    #
    #     # calculate synaptic state
    #     for i in range(num):
    #         if pre_sp[i] > 0.:
    #             idx = pre2syn_slice[i]
    #             ids = pre2syn[idx[0]: idx[1]]
    #             last_sp[ids] = t
    #
    #     # s = st
    #     TT = ((t - last_sp) < T_duration) * T
    #     dsdt = alpha * TT * (1 - s) - beta * s
    #     s += dsdt * dt
    #     syn_st[0] = s
    #
    #     # get post-synaptic values
    #     g = np.zeros(num)
    #     for i in range(num):
    #         idx = post_slice_syn[i]
    #         g[i] += np.sum(s[idx[0]: idx[1]])
    #     post_val = - g_max * g * (post_st[0] - E)
    #     post_st[4] += post_val

    duration = 100.
    t0 = time.time()
    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    for ti in range(tlen):
        t = ts[ti]
        # syn_update_state(syn_state, t, neu_state, neu_state)
        # neu_update_state[512, 512](neu_state)
        neu_update_state[256, 256](neu_state)
        if (ti + 1) * 10 % tlen == 0:
            t1 = time.time()
            print('{} percent {} s'.format((ti + 1) / tlen * 100, t1 - t0))


# HH_model(int(5e7))
HH_model(int(1e8))

