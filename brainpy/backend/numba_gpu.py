# -*- coding: utf-8 -*-

import numba
import numpy
from numba.extending import overload

from .. import profile


@numba.generated_jit(**profile.get_numba_profile())
def normal_like(x):
    if isinstance(x, (numba.types.Integer, numba.types.Float)):
        return lambda x: numpy.random.normal()
    else:
        return lambda x: numpy.random.normal(0., 1.0, x.shape)

