# -*- coding: utf-8 -*-

import numpy as np
import numba as nb
import npbrain as nn
npbrain.profile.set_backend('numba')

neu_state = nn.init_neu_state(1000, ['V', 'm', 'h', 'n'])

# @nn.integrate(method='euler', signature='f8[:](f8[:], f8, f8[:], f8[:])')
# @nb.ve('f8[:](f8[:], f8, f8[:], f8[:])', nopython=True)
# @nb.guvectorize('f8[:](f8[:], f8, f8[:], f8[:])', nopython=True)
# @nb.njit
# def func(a, t, b, c):
#     return a ** 2 + 10 * b - 100 * c

@nb.guvectorize('void(f8[:], f8, f8[:], f8[:])', '(x),(),(x)->(x)', nopython=True)
def func(a, t, b, c):
    for i in range(a.shape[0]):
        c[i] = a[i] ** 2 + 10 * b[i]

# v = func(neu_state.V, 0.1, neu_state.m, bnp.random.random(1000))

func(neu_state.V, 0.1, neu_state.m, np.random.random(1000))

@nb.njit
def f2(neu_state):
    func(neu_state.V, 0.1, neu_state.m, np.random.random(1000))
    # v = func(bnp.random.random(1000), 0.1, bnp.random.random(1000), bnp.random.random(1000))
# func.inspect_types()

f2(neu_state)
