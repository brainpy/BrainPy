# -*- coding: utf-8 -*-

import numpy as np

from . import bk_numba_overload

as_tensor = np.asarray
normal = np.random.normal
reshape = np.reshape
exp = np.exp
sum = np.sum
zeros = np.zeros
ones = np.ones
eye = np.eye
matmul = np.matmul
vstack = np.vstack
arange = np.arange
shape = np.shape
where = np.where


if __name__ == '__main__':
    bk_numba_overload
