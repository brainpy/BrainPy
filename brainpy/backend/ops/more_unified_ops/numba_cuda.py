# -*- coding: utf-8 -*-

from numba import cuda

from brainpy.backend import ops

__all__ = []


@cuda.jit(devicde=True)
def cuda_clip(x, x_min, x_max):
    if x < x_min:
        return x_min
    elif x > x_max:
        return x_max
    else:
        return x


ops.set_buffer('numba-numba',
               clip=cuda_clip
               )
