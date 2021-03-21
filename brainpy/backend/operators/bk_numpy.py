# -*- coding: utf-8 -*-

import numpy as np

__all__ = [
    'as_tensor',
    'normal',
    'reshape',
    'shape',
    'exp',
    'sum',
    'zeros',
    'ones',
    'eye',
    'matmul',
    'vstack',
    'arange',
    'moveaxis',
    'where',
]


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
moveaxis = np.moveaxis
where = np.where


def shape(x):
    size = np.shape(x)
    if len(size) == 0:
        return (1,)
    else:
        return size
