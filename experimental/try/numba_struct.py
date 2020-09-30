# -*- coding: utf-8 -*-

import time
import numba as nb
import numpy as np

dt = 0.02

def try1():
    a = np.zeros(5, dtype=[('p1', 'i8'),
                               ('p2', 'i8'),
                               ('v1', np.float),
                               ('v2', 'float', (10,))])




def define_v1(E_Na=50., g_Na=120., E_K=-77., g_K=36., E_Leak=-54.387, g_Leak=0.03,
              C=1.0, Vth=20.):
    @nb.njit
    def neu_update(pre_st):
        V = pre_st[0]
        m = pre_st[1]
        h = pre_st[2]
        n = pre_st[3]
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

    return neu_update

def define_v2(E_Na=50., g_Na=120., E_K=-77., g_K=36., E_Leak=-54.387, g_Leak=0.03,
                       C=1.0, Vth=20.):
    @nb.njit
    def neu_update(pre_st):
        V = pre_st.V
        m = pre_st.m
        h = pre_st.h
        n = pre_st.n
        input = pre_st.input

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

        pre_st.V[:] = V
        pre_st.m[:] = m
        pre_st.h[:] = h
        pre_st.n[:] = n
        pre_st.sp[:] = V > Vth
        pre_st.input[:] = 10.

    return neu_update


def main():
    num_pre = 1000
    pre_state = np.zeros((6, num_pre, ))
    pre_state[:, 0] = -65.

    neu_update = define_v1()

    duration = 1000.
    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    neu_update(pre_state)
    t0 = time.time()
    for ti in range(1, tlen):
        neu_update(pre_state)
        if (ti + 1) * 20 % tlen == 0:
            t1 = time.time()
            print('{} percent {:.4f} s'.format((ti + 1) / tlen, t1 - t0))



def main2():
    x_dt = np.dtype([('V', np.float64), ('m', np.float64), ('h', np.float64),
                     ('sp', np.float64), ('n', np.float64), ('input', np.float64)], align=True)
    num_pre = 1000

    @nb.njit
    def get_state(num_pre):
        # pre_state = np.recarray((num_pre,), dtype=x_dt)
        pre_state = np.zeros((num_pre,), dtype=x_dt)
        pre_state.V[:] = -65.
        return pre_state

    # st = np.zeros((num_pre, 6))
    # pre_state = np.recarray((num_pre,), dtype=x_dt, buf=st)

    # pre_state = np.recarray((num_pre,), dtype=x_dt)
    # pre_state.V = -65.

    pre_state = get_state(num_pre)

    neu_update = define_v2()

    duration = 1000.
    ts = np.arange(0, duration, dt)
    tlen = len(ts)
    neu_update(pre_state)
    t0 = time.time()
    for ti in range(1, tlen):
        neu_update(pre_state)
        if (ti + 1) * 20 % tlen == 0:
            t1 = time.time()
            print('{} percent {:.4f} s'.format((ti + 1) / tlen, t1 - t0))



main()
# main2()
