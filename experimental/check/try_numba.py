# -*- coding: utf-8 -*-

import time
import numpy as np
import numba as nb


def func1(S, i, st1, st2, st3):
    st1[i] = S[0]
    st2[i] = S[1]
    st3[i] = S[2]


nb_func1 = nb.njit(func1)


num = 3000
state = np.ones((3, 100000))
state1 = np.ones((num, 100000))
state2 = np.ones((num, 100000))
state3 = np.ones((num, 100000))

t0 = time.time()
for i in range(num):
    func1(state, i, state1, state2, state3)
print('numpy time: ', time.time() - t0)


nb_func1(state, 0, state1, state2, state3)
t0 = time.time()
for i in range(1, num):
    nb_func1(state, i, state1, state2, state3)
print('numba time: ', time.time() - t0)


