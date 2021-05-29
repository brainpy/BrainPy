# -*- coding: utf-8 -*-

import numpy as np

from brainpy.backend.ops.necessary_ops import numba_overload
from brainpy.backend.ops.more_unified_ops import numba


__all__ = [
    'normal',
    'exp',
    'sum',
    'shape',
    'as_tensor',
    'zeros',
    'ones',
    'arange',
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
concatenate = np.concatenate
where = np.where
reshape = np.reshape


# necessary ops for dtypes

bool = np.bool_
int = np.int_
int32 = np.int32
int64 = np.int64
float = np.float_
float32 = np.float32
float64 = np.float64


if __name__ == '__main__':
    numba_overload
    numba
