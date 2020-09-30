import time
import numpy as np
# import npbrain as nn
import numba as nb
from numba.typed import List
import arrayfire as af

# import cocos.numerics as af

dt = 0.02


class HH_AF:
    def __init__(self, num, Iext=0.):
        self.E_Na = 50.
        self.g_Na = 120.
        self.E_K = -77.
        self.g_K = 36.
        self.E_Leak = -54.387
        self.g_Leak = 0.03,
        self.C = 1.0
        self.Vth = 20.

        self.num = num
        self.state = af.constant(0, 6, num)
        self.Iext = Iext

    def step(self):
        V = self.state[0]
        m = self.state[1]
        h = self.state[2]
        n = self.state[3]
        input = self.state[5]

        alpha = 0.1 * (V + 40) / (1 - af.exp(-(V + 40) / 10))
        beta = 4.0 * af.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        m = m + dmdt * dt

        alpha = 0.07 * af.exp(-(V + 65) / 20.)
        beta = 1 / (1 + af.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        h = h + dhdt * dt

        alpha = 0.01 * (V + 55) / (1 - af.exp(-(V + 55) / 10))
        beta = 0.125 * af.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        n = n + dndt * dt

        INa = self.g_Na * m ** 3 * h * (V - self.E_Na)
        IK = self.g_K * n ** 4 * (V - self.E_K)
        # IL = self.g_Leak * (V - self.E_Leak)
        # dvdt = (- INa - IK - IL + input) / self.C
        dvdt = (- INa - IK + input) / self.C
        V += dvdt * dt

        self.state[0] = V
        self.state[1] = m
        self.state[2] = h
        self.state[3] = n
        self.state[4] = V > self.Vth
        self.state[5] = self.Iext


class HH_Numpy:
    def __init__(self, num, Iext=0.):
        self.E_Na = 50.
        self.g_Na = 120.
        self.E_K = -77.
        self.g_K = 36.
        self.E_Leak = -54.387
        self.g_Leak = 0.03,
        self.C = 1.0
        self.Vth = 20.

        self.num = num
        self.state = np.zeros((6, num), dtype=np.float32)
        self.Iext = Iext

    def step(self):
        V = self.state[0]
        m = self.state[1]
        h = self.state[2]
        n = self.state[3]
        input = self.state[5]

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

        INa = self.g_Na * m ** 3 * h * (V - self.E_Na)
        IK = self.g_K * n ** 4 * (V - self.E_K)
        IL = self.g_Leak * (V - self.E_Leak)
        dvdt = (- INa - IK - IL + input) / self.C
        V += dvdt * dt

        self.state[0] = V
        self.state[1] = m
        self.state[2] = h
        self.state[3] = n
        self.state[4] = 0.
        idx = np.where(V > self.Vth)[0]
        self.state[4][idx] = 1.
        self.state[5] = self.Iext


def nb_version():
    E_Na = 50.
    g_Na = 120.
    E_K = -77.
    g_K = 36.
    E_Leak = -54.387
    g_Leak = 0.03
    C = 1.0
    Vth = 20.

    num_pre = 1000
    pre_state = np.zeros((6, num_pre))
    pre_state[0] = -65.

    @nb.njit
    def main(pre_st):
        # pre
        V = pre_st[0]
        m = pre_st[1]
        h = pre_st[2]
        n = pre_st[3]
        sp = pre_st[4]
        input = pre_st[5]

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

        pre_st[0] = V
        pre_st[1] = m
        pre_st[2] = h
        pre_st[3] = n
        pre_st[4] = V > Vth
        pre_st[5] = 10.

    duration = 1000.
    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    main(pre_state)
    t0 = time.time()
    for ti in range(1, tlen):
        main(pre_state)
        if (ti + 1) * 20 % tlen == 0:
            t1 = time.time()
            print(' {} percent {:.4f} s'.format((ti + 1) / tlen, t1 - t0))
    print('Numba used: ', t1 - t0)


def np_version():
    num_pre = 1000
    hh = HH_Numpy(num_pre, 10.)

    duration = 1000.
    t0 = time.time()
    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    for ti in range(tlen):
        hh.step()
        if (ti + 1) * 20 % tlen == 0:
            t1 = time.time()
            print('{} percent {:.4f} s'.format((ti + 1) / tlen, t1 - t0))


def af_version():
    E_Na = 50.
    g_Na = 120.
    E_K = -77.
    g_K = 36.
    E_Leak = -54.387
    g_Leak = 0.03
    C = 1.0
    Vth = 20.

    # af.set_backend('cpu')
    af.set_device(1)
    af.info()

    num_pre = 1000
    state = af.constant(0, 6, num_pre)

    def step(state):
        V = state[0]
        m = state[1]
        h = state[2]
        n = state[3]
        input = state[5]

        alpha = 0.1 * (V + 40) / (1 - af.exp(-(V + 40) / 10))
        beta = 4.0 * af.exp(-(V + 65) / 18)
        dmdt = alpha * (1 - m) - beta * m
        m = m + dmdt * dt

        alpha = 0.07 * af.exp(-(V + 65) / 20.)
        beta = 1 / (1 + af.exp(-(V + 35) / 10))
        dhdt = alpha * (1 - h) - beta * h
        h = h + dhdt * dt

        alpha = 0.01 * (V + 55) / (1 - af.exp(-(V + 55) / 10))
        beta = 0.125 * af.exp(-(V + 65) / 80)
        dndt = alpha * (1 - n) - beta * n
        n = n + dndt * dt

        INa = g_Na * m ** 3 * h * (V - E_Na)
        IK = g_K * n ** 4 * (V - E_K)
        IL = g_Leak * (V - E_Leak)
        dvdt = (- INa - IK - IL + input) / C
        V += dvdt * dt

        state[0] = V
        state[1] = m
        state[2] = h
        state[3] = n
        state[4] = V > Vth
        state[5] = 10.

        af.sync()

    duration = 1000.
    t0 = time.time()
    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    for ti in range(tlen):
        step(state)
        if (ti + 1) * 20 % tlen == 0:
            t1 = time.time()
            print('{} percent {:.4f} s'.format((ti + 1) / tlen, t1 - t0))


# nb_version()
# np_version()
if __name__ == '__main__':

    af_version()

