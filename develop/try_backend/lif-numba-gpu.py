# -*- coding: utf-8 -*-

import math
import time

import matplotlib.pyplot as plt
import numpy as np
from numba import cuda

# import brainpy as bp

dt = 0.1


def lif_model(num=20000):
    tau = 100.  # ms
    Vth = 1.  # mV
    Vr = 0.  # mV

    # ST: V-0, spike-1, input-2, Isyn=3

    @cuda.jit
    def update(ST, mon, time_idx, t):
        i = cuda.grid(1)
        if i < num:
            dv = (-ST[0, i] + ST[3, i] + 2 * math.sin(2 * math.pi * t / tau)) / tau
            V = ST[0, i] + dv * dt
            if V >= Vth:
                ST[1, i] = 1.
                ST[0, i] = Vr
            else:
                ST[1, i] = 0.
                ST[0, i] = V
            # ST[3, i] = 0.
            mon[time_idx, i] = ST[1, i]

    duration = 5000.
    ts = np.arange(0, duration, dt)
    tlen = len(ts)

    # ST: V-0, spike-1, input-2, Isyn=3
    inputs = np.linspace(2., 4., num)
    neu_state = np.zeros((4, num))
    neu_state[0] = Vr
    neu_state[3] = inputs
    mon = np.zeros((int(duration / dt), num))

    threads_per_block = 1024
    blocks_per_grid = math.ceil(num / threads_per_block)

    def no_stream():
        neu_state_cuda = cuda.to_device(neu_state)
        mon_cuda = cuda.to_device(mon)

        update[blocks_per_grid, threads_per_block](neu_state_cuda, mon_cuda, 0, ts[0])
        cuda.synchronize()
        t0 = time.time()
        for ti in range(1, tlen):
            update[blocks_per_grid, threads_per_block](neu_state_cuda, mon_cuda, ti, ts[ti])
            cuda.synchronize()
            if (ti + 1) * 10 % tlen == 0:
                t1 = time.time()
                print('{} percent {} s'.format((ti + 1) / tlen * 100, t1 - t0))
        mon_cuda.to_host()

        indices, times = bp.measure.raster_plot(mon, ts)
        plt.plot((times % tau) / tau, inputs[indices], ',')

        # plt.plot(mon[:, 0])
        plt.show()

    no_stream()


def lif_model_v2(num=20000, duration=5000.):
    tau = 100.  # ms
    Vth = 1.  # mV
    Vr = 0.  # mV

    # ST: V-0, spike-1, input-2, Isyn=3

    def update(ST, t):
        i = cuda.grid(1)
        if i < num:
            dv = (-ST[0, i] + ST[3, i] + 2 * math.sin(2 * math.pi * t / tau)) / tau
            V = ST[0, i] + dv * dt
            if V >= Vth:
                ST[1, i] = 1.
                ST[0, i] = Vr
            else:
                ST[1, i] = 0.
                ST[0, i] = V

    gpu_update = cuda.jit(update)

    ts = np.arange(0, duration, dt)
    tlen = len(ts)

    # ST: V-0, spike-1, input-2, Isyn=3
    inputs = np.linspace(2., 4., num)

    threads_per_block = 1024
    blocks_per_grid = math.ceil(num / threads_per_block)

    def with_stream():
        neu_state = np.zeros((4, num))
        neu_state[0] = Vr
        neu_state[3] = inputs

        stream = cuda.stream()
        neu_state_cuda = cuda.to_device(neu_state, stream)

        gpu_update[blocks_per_grid, threads_per_block, stream](neu_state_cuda, ts[0])
        stream.synchronize()
        t0 = time.time()
        for ti in range(1, tlen):
            gpu_update[blocks_per_grid, threads_per_block, stream](neu_state_cuda, ts[ti])
            stream.synchronize()
            if (ti + 1) * 10 % tlen == 0:
                t1 = time.time()
                print('{} percent {} s'.format((ti + 1) / tlen * 100, t1 - t0))

    def cpu_run():
        neu_state = np.zeros((4, num))
        neu_state[0] = Vr
        neu_state[3] = inputs

        stream = cuda.stream()
        neu_state_cuda = cuda.to_device(neu_state, stream)

        gpu_update[blocks_per_grid, threads_per_block](neu_state_cuda, ts[0])
        stream.synchronize()
        t0 = time.time()
        for ti in range(1, tlen):
            gpu_update[blocks_per_grid, threads_per_block, stream](neu_state_cuda, ts[ti])
            stream.synchronize()
            if (ti + 1) * 10 % tlen == 0:
                t1 = time.time()
                print('{} percent {} s'.format((ti + 1) / tlen * 100, t1 - t0))

    print('CPU')
    # cpu_run()
    # no_stream()
    print('GPU')
    with_stream()


# lif_model(int(2e3))
lif_model_v2(int(2e6))
