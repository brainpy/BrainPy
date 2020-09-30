# -*- coding: utf-8 -*-

import time

import numpy as np
from numba import njit
from numba.core import types
from numba.typed.typeddict import Dict, _getitem, _contains
from numba.extending import overload_method


d = Dict.empty(key_type=types.unicode_type, value_type=types.int64)
d['V'] = 0
d['m'] = 1
d['h'] = 2
d['n'] = 3
d['not_ref'] = 4
d['above_th'] = 5
d['spike'] = 6
d['spike_time'] = 7
d['Isyn'] = 8


@overload_method(types.Array, '__getitem__')
def array_take(arr, indices):
    if isinstance(indices, types.unicode_type):
        def take_impl(arr, indices):
            return arr[d[indices]]
        return take_impl


def try_HH_model_without_dict(num=20000):
    E_Na = 50.
    g_Na = 120.
    E_K = -77.
    g_K = 36.
    E_Leak = -54.387
    g_Leak = 0.03
    C = 1.0
    Vr = -65.
    Vth = 20.
    dt = 0.02

    neu_state = np.zeros((4 + 5, num))
    neu_state[0] = Vr
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
    def neu_update_state(st, t):
        V = st[0]
        m = st[1]
        h = st[2]
        n = st[3]

        m = int_m(m, t, V)
        h = int_h(h, t, V)
        n = int_n(n, t, V)

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


def try_HH_model_with_dict(num=20000):
    E_Na = 50.
    g_Na = 120.
    E_K = -77.
    g_K = 36.
    E_Leak = -54.387
    g_Leak = 0.03
    C = 1.0
    Vr = -65.
    Vth = 20.
    dt = 0.02

    neu_state = np.zeros((4 + 5, num))
    neu_state[0] = Vr
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
        above_threshold = (neu_state['V'] >= vth).astype(np.float64)
        prev_above_th = neu_state['above_th']
        spike_st = above_threshold * (1. - prev_above_th)
        spike_idx = np.where(spike_st > 0.)[0]
        neu_state['above_th'] = above_threshold
        neu_state['spike'] = spike_st
        neu_state['spike_time'][spike_idx] = t
        return spike_idx

    @njit
    def neu_update_state(st, t):
        V = st['V']
        m = st['m']
        h = st['h']
        n = st['n']

        m = int_m(m, t, V)
        h = int_h(h, t, V)
        n = int_n(n, t, V)

        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + st['Isyn']) / C
        V += dvdt * dt

        st['V'] = V
        st['m'] = m
        st['h'] = h
        st['n'] = n
        st['Isyn'][:] = 0.
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
    # try_HH_model_without_dict()
    try_HH_model_with_dict()
