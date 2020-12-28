# -*- coding: utf-8 -*-

import math
import time
import matplotlib.pyplot as plt
from numba import cuda

import numpy as np

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




    @cuda.jit
    def neu_update_state(st, mon, time_idx):
        i = cuda.grid(1)

        if i < st.shape[1]:
            # V = st[0]
            # m = st[1]
            # h = st[2]
            # n = st[3]
            # Isyn = st[4]

            alpha = 0.1 * (st[0, i] + 40) / (1 - math.exp(-(st[0, i] + 40) / 10))
            beta = 4.0 * math.exp(-(st[0, i] + 65) / 18)
            dmdt = alpha * (1 - st[1, i]) - beta * st[1, i]
            st[1, i] = st[1, i] + dmdt * dt

            alpha = 0.07 * math.exp(-(st[0, i] + 65) / 20.)
            beta = 1 / (1 + math.exp(-(st[0, i] + 35) / 10))
            dhdt = alpha * (1 - st[2, i]) - beta * st[2, i]
            st[2, i] = st[2, i] + dhdt * dt

            alpha = 0.01 * (st[0, i] + 55) / (1 - math.exp(-(st[0, i] + 55) / 10))
            beta = 0.125 * math.exp(-(st[0, i] + 65) / 80)
            dndt = alpha * (1 - st[3, i]) - beta * st[3, i]
            st[3, i] = st[3, i] + dndt * dt

            INa = g_Na * st[1, i] ** 3 * st[2, i] * (st[0, i] - E_Na)
            IK = g_K * st[3, i] ** 4 * (st[0, i] - E_K)
            IL = g_Leak * (st[0, i] - E_Leak)
            Icur = - INa - IK - IL + st[5, i]
            dvdt = (Icur + st[4, i]) / C
            st[0, i] += dvdt * dt

            if i == 0:
                mon[time_idx] = st[0, i]


    duration = 100.
    ts = np.arange(0, duration, dt)
    tlen = len(ts)

    # ['V', 'm', 'h', 'n', Isyn]
    neu_state = np.zeros((6, num))
    neu_state[0] = Vr
    neu_state[5] = 10.
    mon = np.zeros((int(duration / dt)))

    threads_per_block = 1024
    blocks_per_grid = math.ceil(num / threads_per_block)

    def use_stream():
        stream = cuda.stream()
        neu_state_cuda = cuda.to_device(neu_state, stream)
        mon_cuda = cuda.to_device(mon, stream)

        neu_update_state[blocks_per_grid, threads_per_block, stream](neu_state_cuda, mon_cuda, 0)
        stream.synchronize()
        t0 = time.time()
        for ti in range(1, tlen):
            neu_update_state[blocks_per_grid, threads_per_block, stream](neu_state_cuda, mon_cuda, ti)
            stream.synchronize()
            if (ti + 1) * 10 % tlen == 0:
                t1 = time.time()
                print('{} percent {} s'.format((ti + 1) / tlen * 100, t1 - t0))
        mon_cuda.to_host()

        plt.plot(mon)
        plt.show()

    def no_stream():
        neu_state_cuda = cuda.to_device(neu_state)
        mon_cuda = cuda.to_device(mon)

        neu_update_state[blocks_per_grid, threads_per_block](neu_state_cuda, mon_cuda, 0)
        cuda.synchronize()
        t0 = time.time()
        for ti in range(1, tlen):
            neu_update_state[blocks_per_grid, threads_per_block](neu_state_cuda, mon_cuda, ti)
            cuda.synchronize()
            if (ti + 1) * 10 % tlen == 0:
                t1 = time.time()
                print('{} percent {} s'.format((ti + 1) / tlen * 100, t1 - t0))
        mon_cuda.to_host()

        plt.plot(mon)
        plt.show()


    def no_stream_to_host():
        neu_state_cuda = cuda.to_device(neu_state)
        mon_cuda = cuda.to_device(mon)

        neu_update_state[blocks_per_grid, threads_per_block](neu_state_cuda, mon_cuda, 0)
        cuda.synchronize()
        t0 = time.time()
        for ti in range(1, tlen):
            neu_update_state[blocks_per_grid, threads_per_block](neu_state_cuda, mon_cuda, ti)
            cuda.synchronize()
            if (ti + 1) * 10 % tlen == 0:
                t1 = time.time()
                print('{} percent {} s'.format((ti + 1) / tlen * 100, t1 - t0))
                mon_cuda.to_host()
        mon_cuda.to_host()

        plt.plot(mon)
        plt.show()

    # use_stream()
    # t0 = time.time()
    # no_stream()

    no_stream_to_host()


HH_model(int(1e5))
# HH_model(int(1e8))

