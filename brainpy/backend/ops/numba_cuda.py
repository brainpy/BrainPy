# -*- coding: utf-8 -*-


import math
import numpy as np

from . import numba_overload


# necessary ops for integrators
normal = np.random.normal
sum = np.sum
shape = np.shape
exp = math.exp

# necessary ops for dynamics simulation
as_tensor = np.asarray
zeros = np.zeros
ones = np.ones
arange = np.arange
vstack = np.vstack
where = np.where
unsqueeze = np.expand_dims
squeeze = np.squeeze


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
