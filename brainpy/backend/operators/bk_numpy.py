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
]


as_tensor = np.asarray
normal = np.random.normal
reshape = np.reshape
shape = np.shape
exp = np.exp
sum = np.sum
zeros = np.zeros
ones = np.ones
eye = np.eye
matmul = np.matmul
vstack = np.vstack
arange = np.arange

