
import numpy as np
import numba as nb
import npbrain.all as nn
from timeit import timeit
import time


def run(f, num=10, *args):
    t0 = time.time()
    for _ in range(num):
        f(*args)
    t1 = time.time()
    return t1 - t0


N = pre_num = post_num = 5000
pre_indexes, post_indexes, pre_anchors = nn.conn.fixed_prob(N, N, 0.1)
conn_mat = np.zeros((N, N))
conn_mat[pre_indexes, post_indexes] = 1.

pre_spike = np.zeros(N)
spike_idx = np.random.randint(0, N, 1000)
pre_spike[spike_idx] = 1.



# def np_syn1():
#     return bnp.dot(pre_spike, conn_mat)
# @nb.njit
# def nb_syn1():
#     return bnp.dot(pre_spike, conn_mat)
# if __name__ == '__main__':
#     num = 1000
#     num = 2000
#     num = 1000
#     print('np_syn : ', run(np_syn1, num))
#     print('nb_syn : ', run(nb_syn1, num))


# pre2syn, post2syn = nn.conn.correspondence(N, N, pre_indexes, post_indexes)
# pre2syn = [v for _, v in sorted(pre2syn.items())]
# post2syn = [v for _, v in sorted(post2syn.items())]
#
# def np_syn2():
#     syn_val = bnp.zeros((len(post_indexes),))
#     for i_ in spike_idx:
#         syn_val[pre2syn[i_]] = 1.
#     post_val = bnp.zeros(N)
#     for i, j_ in enumerate(post2syn):
#         post_val[i] = bnp.sum(syn_val[j_])
# if __name__ == '__main__':
#     num = 100
#     print('np_syn2 : ', timeit("np_syn2()", setup="from __main__ import np_syn2", number=num))
#     print('np_syn2 : ', run(np_syn2, num))



# pre2post = [bnp.where(conn_mat[i] > 0)[0] for i in range(N)]
# post2pre = (bnp.where(conn_mat[:, i] > 0)[0] for i in range(N))
# def np_syn3():
#     post_val = bnp.zeros(post_num)
#     for i in range(post_num):
#         post_val[i] = bnp.sum(pre_spike[post2pre[i]])
#     return post_val
# @nb.njit
# def nb_syn3():
#     post_val = bnp.zeros(post_num)
#     for i in range(post_num):
#         idx = post2pre[i]
#         post_val[i] = bnp.sum(pre_spike[idx])
#     return post_val
#
# if __name__ == '__main__':
#     num = 1000
#     print('Numpy: ', timeit("np_syn3()", setup="from __main__ import np_syn3", number=num))
#     print('Numba: ', timeit("nb_syn3()", setup="from __main__ import nb_syn3", number=num))


pre2post = list([list(np.where(conn_mat[i] > 0)[0]) for i in range(N)])
def np_syn4():
    post_val = np.zeros(post_num)
    spike_idx = np.where(pre_spike > 0)[0]
    for i in spike_idx:
        post_idx = pre2post[i]
        post_val[post_idx] += 1
    return post_val
# pre2post_nb = types.Tuple(pre2post)
@nb.njit
def nb_syn4(pre2post_nb):
    post_val = np.zeros(post_num)
    spike_idx = np.where(pre_spike > 0)[0]
    for i in spike_idx:
        post_idx = pre2post_nb[i]
        for j in post_idx:
            post_val[j] += 1
    return post_val

if __name__ == '__main__':
    num = 1
    print('Numpy: ', run(np_syn4, num))
    print('Numba: ', run(nb_syn4, num, pre2post))



# ii = 0
# post_indexes = []
# pre_anchors = []
# for pre_indexes in range(N):
#     post_indexes = bnp.where(conn_mat[pre_indexes] > 0)[0]
#     post_indexes.extend(post_indexes)
#     length = len(post_indexes)
#     pre_anchors.append([ii, ii + length])
#     ii += length
# post_indexes = bnp.array(post_indexes)
# pre_anchors = bnp.array(pre_anchors)
def np_syn5():
    post_val = np.zeros(post_num)
    spike_idx = np.where(pre_spike > 0)[0]
    for i_ in spike_idx:
        start, end = pre_anchors[:, i_]
        post_idx = post_indexes[start: end]
        post_val[post_idx] += 1
    return post_val
@nb.njit
def nb_syn5():
    post_val = np.zeros(post_num)
    spike_idx = np.where(pre_spike > 0)[0]
    for i_ in spike_idx:
        start, end = pre_anchors[:, i_]
        post_idx = post_indexes[start: end]
        post_val[post_idx] += 1
    return post_val


if __name__ == '__main__':
    # np_res = np_syn5()
    # nb_res = nb_syn5()
    # assert bnp.allclose(np_res, nb_res)

    num = 1000
    print('np_syn5 : ', run(np_syn5, num))
    print('nb_syn5 : ', run(nb_syn5, num))




