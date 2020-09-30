# -*- coding: utf-8 -*-

import numpy as np
import numba as nb
import torch
import time


@torch.jit.script
class Test:
    def __init__(self):
        pass

    def calc(self, input: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.exp(input)


@torch.jit.script
def calc(a):
    return 1.0 / torch.exp(a)


@nb.njit
def nb_calc(a):
    return 1.0 / np.exp(a)


# print('Class ...')
#
# for i in range(2):
#     test = Test()
#     start = time.time()
#     a = torch.zeros(6000, 200)
#     for j in range(1000):
#         a = test.calc(a)
#     print(time.time() - start)
#
print('Function ...')
torch.set_num_threads(1)

for i in range(5):
    start = time.time()
    a = torch.zeros(6000, 200)
    for j in range(1000):
        a = calc(a)
    print(time.time() - start)

print('Numba ...')

for i in range(5):
    start = time.time()
    a = np.zeros((6000, 200))
    for j in range(1000):
        a = nb_calc(a)
    print(time.time() - start)
