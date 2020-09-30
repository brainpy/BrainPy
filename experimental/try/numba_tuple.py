# -*- coding: utf-8 -*-

from collections import namedtuple

import numpy as np

from numba import njit


MyRec = namedtuple("MyRec", "indices,values,v2")


@njit
def make(ids, vals):
    return MyRec(indices=ids, values=vals, v2=np.array([0]))


@njit
def foo(rec):
    rec.v2[0] = 1
    return rec.indices + rec.values


@njit
def bar():
    a = np.arange(10)
    b = np.arange(10)
    c = make(a, b)
    return foo(c)


myrec = make(np.arange(10).reshape((2, 5)), np.zeros((1, 10)))
print(myrec)
# print(foo(myrec))
# print(bar())
print(myrec.v2)
print(myrec.indices[myrec.v2])

# myrec.a
