# -*- coding: utf-8 -*-


from __future__ import print_function

import math
import numpy
from numba import cuda

num = 100000
CONST1D = numpy.arange(num, dtype=numpy.float64) / 2.
CONST2D = numpy.asfortranarray(numpy.arange(100, dtype=numpy.int32).reshape(10, 10))
CONST3D = ((numpy.arange(5 * 5 * 5, dtype=numpy.complex64).reshape(5, 5, 5) + 1j) / 2j)


@cuda.jit('void(float64[:])')
def cuconst(A):
    C = cuda.const.array_like(CONST1D)
    i = cuda.grid(1)
    A[i] = C[i]


@cuda.jit('void(int32[:,:])')
def cuconst2d(A):
    C = cuda.const.array_like(CONST2D)
    i, j = cuda.grid(2)
    A[i, j] = C[i, j]


@cuda.jit('void(complex64[:,:,:])')
def cuconst3d(A):
    C = cuda.const.array_like(CONST3D)
    i = cuda.threadIdx.x
    j = cuda.threadIdx.y
    k = cuda.threadIdx.z
    A[i, j, k] = C[i, j, k]


def try_const_array():
    assert '.const' in cuconst.ptx
    A = numpy.empty_like(CONST1D)
    cuconst[math.ceil(num / 256), 256](A)
    assert numpy.all(A == CONST1D)


def try_const_array_2d():
    assert '.const' in cuconst2d.ptx
    A = numpy.empty_like(CONST2D, order='C')
    cuconst2d[(2, 2), (5, 5)](A)
    assert numpy.all(A == CONST2D)


def try_const_array_3d():
    assert '.const' in cuconst3d.ptx
    A = numpy.empty_like(CONST3D, order='F')
    cuconst3d[1, (5, 5, 5)](A)
    assert numpy.all(A == CONST3D)


try_const_array()
