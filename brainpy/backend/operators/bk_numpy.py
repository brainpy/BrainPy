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
    'arange',

    'vstack',
    'where',
    'unsqueeze',
    'squeeze',
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
arange = np.arange

vstack = np.vstack
where = np.where
unsqueeze = np.expand_dims
squeeze = np.squeeze
