# -*- coding: utf-8 -*-

# NumPy
import numpy as np

import brainpy as bp

bp.ops.set_buffer('numpy', clip=np.clip)

# PyTorch
try:
    import torch

    bp.ops.set_buffer('pytorch', clip=torch.clamp)

except ModuleNotFoundError:
    pass

# TensorFlow
try:
    import tensorflow as tf

    bp.ops.set_buffer('tensorflow', clip=tf.clip_by_value)

except ModuleNotFoundError:
    pass

# Numba
try:
    import numba as nb
    from numba import cuda

    @nb.njit
    def nb_clip(x, x_min, x_max):
        x = np.maximum(x, x_min)
        x = np.minimum(x, x_max)
        return x


    bp.ops.set_buffer('numba', clip=nb_clip)
    bp.ops.set_buffer('numba-parallel', clip=nb_clip)


    @cuda.jit(devicde=True)
    def cuda_clip(x, x_min, x_max):
        if x < x_min: return x_min
        elif x > x_max: return x_max
        else: return x


    bp.ops.set_buffer('numba-numba', clip=nb_clip)
except ModuleNotFoundError:
    pass
