import time
import numpy as np
import npbrain.all as nn
from numba import njit


def run(f, num=10, *args):
    t0 = time.time()
    for _ in range(num):
        f(*args)
    t1 = time.time()
    return t1 - t0

np.random.seed(1234)


dt = 0.1
num_pre = 1000
num_post = 2000
pre_indexes, post_indexes, pre_anchors = nn.conn.format_connection(
    # {'method': 'fixed_prob', 'prob': 0.2},
    {'method': 'fixed_prob', 'prob': 0.8},
    num_pre, num_post)
# post_indexes, pre_anchors = nn.conn.format_pre2post(conn_mat, num_pre)
conn_mat = np.zeros((num_pre, num_post))
conn_mat[pre_indexes, post_indexes] = 1.

# conn_mat2 = bnp.zeros((num_pre, num_post))
# for i in range(num_pre):
#     start, end = pre_anchors[:, i]
#     post_idx = post_indexes[start: end]
#     conn_mat2[i, post_idx] = 1.
# assert bnp.allclose(conn_mat, conn_mat2)


pre_spike = np.zeros(num_pre)
spike_idx = np.random.randint(0, num_pre, int(num_pre * 0.2))
spike_idx = np.array(list(set(spike_idx)))
pre_spike[spike_idx] = 1.

tau_decay = 2.
int_f = lambda s, t: s + (-s / tau_decay) * dt
int_f_nb = njit(lambda s, t: s + (-s / tau_decay) * dt)
syn_state_1d = np.ones(len(post_indexes))
syn_state_2d = np.ones((num_pre, num_post)) * conn_mat




def ampa1_a1_np():
    post_val = np.zeros(num_post)
    for i in spike_idx:
        start, end = pre_anchors[:, i]
        post_idx = post_indexes[start: end]
        post_val[post_idx] += 1
    syn_val = int_f(syn_state_1d, None)
    for i in range(num_pre):
        start, end = pre_anchors[:, i]
        post_idx = post_indexes[start: end]
        post_val[post_idx] += syn_val[start: end]
    return post_val
def ampa1_a2_np():
    spikes_in_syns = pre_spike[np.newaxis].T * conn_mat
    syn_val = int_f(syn_state_2d, None)
    syn_val += spikes_in_syns
    post_val = np.sum(syn_val, axis=0)
    return post_val
def ampa1_a3_np():
    syn_val = np.zeros((num_pre, num_post))
    for i in spike_idx:
        start, end = pre_anchors[:, i]
        post_idx = post_indexes[start: end]
        syn_val[i, post_idx] += 1
    syn_val += int_f(syn_state_2d, None)
    post_val = np.sum(syn_val, axis=0)
    return post_val
@njit
def ampa1_a1_nb():
    post_val = np.zeros(num_post)
    for i in spike_idx:
        start, end = pre_anchors[:, i]
        post_idx = post_indexes[start: end]
        post_val[post_idx] += 1
    syn_val = int_f_nb(syn_state_1d, 0)
    for i in range(num_pre):
        start, end = pre_anchors[:, i]
        post_idx = post_indexes[start: end]
        post_val[post_idx] += syn_val[start: end]
    return post_val
@njit
def ampa1_a2_nb():
    spikes_in_syns = pre_spike.reshape((-1, 1)) * conn_mat
    syn_val = int_f_nb(syn_state_2d, 0)
    syn_val += spikes_in_syns
    post_val = np.sum(syn_val, axis=0)
    return post_val


if __name__ == '__main__1':
    a = ampa1_a1_np()
    b = ampa1_a2_np()
    c = ampa1_a3_np()
    assert np.allclose(a, b)
    assert np.allclose(b, c)


if __name__ == '__main__':
    print('ampa1_a1_np =', run(ampa1_a1_np, 100))
    print('ampa1_a2_np =', run(ampa1_a2_np, 100))
    print('ampa1_a3_np =', run(ampa1_a3_np, 100))
    print('ampa1_a1_nb =', run(ampa1_a1_nb, 100))
    print('ampa1_a2_nb =', run(ampa1_a2_nb, 100))

