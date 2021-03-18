# -*- coding: utf-8 -*-

import numba
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
outer = np.outer
matmul = np.matmul
vstack = np.vstack
arange = np.arange
shape = np.shape

#
# @numba.njit
# def shape(x):
#     size = np.shape(x)
#     if len(size) == 0:
#         return (1,)
#     else:
#         return size


@numba.generated_jit(fastmath=True, nopython=True, nogil=True)
def normal_like(x):
    if isinstance(x, (numba.types.Integer, numba.types.Float)):
        return lambda x: np.random.normal()
    else:
        return lambda x: np.random.normal(0., 1.0, x.shape)


if __name__ == '__main__':
    bk_numba_overload
