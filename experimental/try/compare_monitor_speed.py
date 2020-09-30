
import time
import numpy as np
import numba as nb


def compare1():

    def step(mon_state, neu_state, i):
        neu_state[:] = np.random.random(width)
        mon_state[i] = neu_state

    step_nb = nb.njit(step)
    step_pa = nb.njit(step, parallel=True)


    length, width = 10000, 1000
    length, width = 10, 10

    mon_st = np.zeros((length, width))
    neu_st = np.ones(width)
    t0 = time.time()
    for i in range(length):
        step(mon_st, neu_st, i)
    print('Numpy used ', time.time() - t0)
    print(mon_st)


    mon_st = np.zeros((length, width))
    neu_st = np.random.random(width)
    step_nb(mon_st, neu_st, 0)
    t0 = time.time()
    for i in range(1, length):
        step_nb(mon_st, neu_st, i)
    print('Numba used ', time.time() - t0)
    print(mon_st)


    # mon_st = np.zeros((length, width))
    # neu_st = np.random.random(width)
    # step_pa(mon_st, neu_st, 0)
    # t0 = time.time()
    # for i in range(1, length):
    #     step_pa(mon_st, neu_st, i)
    # print('Numba parallel used ', time.time() - t0)


def compare2():
    def step(mon_state, neu_state, i):
        mon_state[i] = neu_state

    def step_nb(mon_state, neu_state, i):
        for j in range(neu_state.shape[0]):
            mon_state[i, j] = neu_state[j]

    length = 10000
    width = 1000

    mon_st = np.zeros((length, width))
    neu_st = np.random.random(width)
    t0 = time.time()
    for i in range(length):
        step(mon_st, neu_st, i)
    print('Numpy used ', time.time() - t0)

    mon_st = np.zeros((length, width))
    neu_st = np.random.random(width)
    t0 = time.time()
    for i in range(length):
        step_nb(mon_st, neu_st, i)
    print('Numba used ', time.time() - t0)


compare1()
# compare2()
