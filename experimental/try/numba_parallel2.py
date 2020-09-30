# -*- coding: utf-8 -*-

import time
import numpy as np
import npbrain as nn
import numba as nb
# from numba import range
parallel = True

dt = 0.1
duration = 10 * 1000
num = 2000
sparseness = 0.1
J = .1
tau = 20.
Vr = 10.
Vth = 20.
noise = 1.
ref = 2.
delay = 2.

pre_ids, post_ids, anchors = nn.connect.fixed_prob(num, num, sparseness, False)

neu_state = nn.init_neu_state(1, num)
neu_state[0] = Vr

syn_state = nn.init_syn_state(delay, num, num, len(pre_ids))


@nb.njit(nogil=True, parallel=parallel, fastmath=True)
def neu_update_state(neu_state, t):
    for idx in nb.prange(num):
        if (t - neu_state[-2, idx]) > ref:
            v = neu_state[0, idx]
            Isyn = neu_state[-1, idx]
            v = v + (-v + Vr + Isyn) / tau * dt
            if v >= Vth:
                neu_state[-5, idx] = 0.  # refractory state
                neu_state[-3, idx] = 1.  # spike state
                neu_state[-2, idx] = t  # spike time
                v = Vr
            else:
                neu_state[-5, idx] = 1.
                neu_state[-3, idx] = 0.
            neu_state[0, idx] = v  # membrane potential
        else:
            neu_state[-5, idx] = 0.
            neu_state[-3, idx] = 0.


@nb.njit(nogil=True, parallel=parallel, fastmath=True)
def syn_update_state(syn_state, t, delay_idx):
    g = np.zeros(num)
    pre_state = syn_state[0]
    for pre_idx in nb.prange(num):
        if pre_state[-1, pre_idx] > 0.:
            idx = anchors[:, pre_idx]
            post_idx = post_ids[idx[0]: idx[1]]
            for idx in post_idx:
                g[idx] += J
    # update `conductance`
    syn_state[1][delay_idx] = g


@nb.njit(nogil=True, parallel=parallel, fastmath=True)
def syn_output_synapse(syn_state, output_idx, post_neu_state):
    g_val = syn_state[1][output_idx]
    for idx in nb.prange(num):
        post_neu_state[0, idx] += g_val[idx] * post_neu_state[-5, idx]


@nb.njit(nogil=False, parallel=False, fastmath=True)
def collect_spike(syn_state, pre_neu_state):
    state = syn_state[0]
    for i in range(num):
        state[-1, i] = pre_neu_state[-3, i]

#
# syn_update_state(syn_state, 0., 10)
# syn_output_synapse(syn_state, 0, neu_state)
# neu_update_state(neu_state, 0.)
# print(neu_update_state.parallel_diagnostics(level=4))
# print(syn_update_state.parallel_diagnostics(level=4))
# print(syn_output_synapse.parallel_diagnostics(level=4))


t0 = time.time()
delay_len = int(delay / dt) + 1
delay_idx = delay_len - 1
output_idx = 0
for t in np.arange(0, duration, dt):
    neu_state[-1] = 25.
    collect_spike(syn_state, neu_state)
    syn_update_state(syn_state, t, delay_idx)
    syn_output_synapse(syn_state, output_idx, neu_state)
    neu_update_state(neu_state, t)
    delay_idx = (delay_idx + 1) % delay_len
    output_idx = (output_idx + 1) % delay_len
t1 = time.time()
print('Time : ', t1 - t0)

