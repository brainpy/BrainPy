# -*- coding: utf-8 -*-

import numpy as np
import numba as nb
import time

# @nb.cfunc('f8[:](f8[:], f8[:])')

def f(m, V):
    alpha = 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
    beta = 4.0 * np.exp(-(V + 65) / 18)
    return alpha * (1 - m) - beta * m


f_jit = nb.njit(f)
f_cfunc = nb.cfunc('f8[:](f8[:], f8[:])')(f)


a = np.zeros(1000)
b = np.zeros(1000)

t0 = time.time()
for _ in range(1000):
    f_jit(a, b)
t1 = time.time()
print("f_jit : ", t1 - t0)


t0 = time.time()
for _ in range(1000):
    f_cfunc(a, b)
t1 = time.time()
print("f_cfunc : ", t1 - t0)

