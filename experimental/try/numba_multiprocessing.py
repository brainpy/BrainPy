import numpy as np
import time
import multiprocessing
from numba import njit


def calc_sum(a):
    a_cum = 0.
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            for n in range(a.shape[2]):
                a_cum += a[i, j, n]
    return a_cum


def calc_sum2(a):
    def f():
        a_cum = 0.
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for n in range(a.shape[2]):
                    a_cum += a[i, j, n]
        return a_cum
    return f()


jit_sum = njit(calc_sum)


def jit_sum2(a):
    @njit
    def f():
        a_cum = 0.
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                for n in range(a.shape[2]):
                    a_cum += a[i, j, n]
        return a_cum
    return f()


if __name__ == '__main__1':
    a = np.random.random((10, 20, 30))
    jit_sum(a)

if __name__ == '__main__1':
    a = np.random.random((100, 200, 300))
    t0 = time.time()
    r21 = jit_sum(a)
    print('numba : ', time.time() - t0)

    t0 = time.time()
    r22 = calc_sum(a)
    print('numpy : ', time.time() - t0)
    assert r21 == r22


if __name__ == '__main__1':
    a = np.random.random((200, 300, 400))
    b = np.random.random((200, 300, 400))
    c = np.random.random((200, 300, 400))

    t0 = time.time()
    r11 = jit_sum(a)
    r12 = jit_sum(b)
    r13 = jit_sum(c)
    print('numba - single : ', time.time() - t0)

    t0 = time.time()
    pool = multiprocessing.Pool(processes=3)
    r = pool.map(jit_sum, [a, b, c])
    pool.close()
    pool.join()
    print('numba - multi : ', time.time() - t0)

    t0 = time.time()
    r21 = calc_sum(a)
    r22 = calc_sum(b)
    r23 = calc_sum(c)
    print('numpy : ', time.time() - t0)

    assert r11 == r21 == r[0] and \
           r12 == r22 == r[1] and \
           r13 == r23 == r[2]


if __name__ == '__main__':
    a = np.random.random((200, 300, 400))
    b = np.random.random((200, 300, 400))
    c = np.random.random((200, 300, 400))

    t0 = time.time()
    r11 = jit_sum2(a)
    r12 = jit_sum2(b)
    r13 = jit_sum2(c)
    print('numba - single : ', time.time() - t0)

    t0 = time.time()
    pool = multiprocessing.Pool(processes=3)
    r = pool.map(jit_sum2, [a, b, c])
    pool.close()
    pool.join()
    print('numba - multi : ', time.time() - t0)

    t0 = time.time()
    r21 = calc_sum2(a)
    r22 = calc_sum2(b)
    r23 = calc_sum2(c)
    print('numpy : ', time.time() - t0)

    assert r11 == r21 == r[0] and \
           r12 == r22 == r[1] and \
           r13 == r23 == r[2]
