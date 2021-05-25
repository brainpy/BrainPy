# -*- coding: utf-8 -*-

import numpy as np

from .more_unified_ops.numpy_ import *

__all__ = [
    'normal',
    'shape',
    'exp',
    'sum',

    'as_tensor',
    'zeros',
    'ones',
    'arange',
    'vstack',
    'concatenate',
    'where',
    'reshape',

    'bool',
    'int',
    'int32',
    'int64',
    'float',
    'float32',
    'float64'
]

# necessary ops for integrators
normal = np.random.normal
sum = np.sum
shape = np.shape
exp = np.exp

# necessary ops for dynamics simulation
as_tensor = np.asarray
zeros = np.zeros
ones = np.ones
arange = np.arange
vstack = np.vstack
concatenate = np.concatenate
where = np.where
# unsqueeze = np.expand_dims
# squeeze = np.squeeze
reshape = np.reshape


# necessary ops for dtypes

bool = np.bool_
int = np.int_
int32 = np.int32
int64 = np.int64
float = np.float_
float32 = np.float32
float64 = np.float64
