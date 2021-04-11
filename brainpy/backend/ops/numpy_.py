# -*- coding: utf-8 -*-

import numpy as np

__all__ = [
    'normal',
    'shape',
    'exp',
    'sum',
    'matmul',

    'as_tensor',
    'zeros',
    'ones',
    'arange',
    'vstack',
    'where',
    'unsqueeze',
    'squeeze',
]

# necessary ops for integrators
normal = np.random.normal
sum = np.sum
shape = np.shape
exp = np.exp
matmul = np.matmul

# necessary ops for dynamics simulation
as_tensor = np.asarray
zeros = np.zeros
ones = np.ones
arange = np.arange
vstack = np.vstack
where = np.where
unsqueeze = np.expand_dims
squeeze = np.squeeze
