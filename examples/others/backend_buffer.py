# -*- coding: utf-8 -*-

import brainpy as bp

# NumPy
import numpy as np
bp.backend.set_buffer('numpy', {'clip': np.clip})

# PyTorch
try:
    import torch

    bp.backend.set_buffer('pytorch', {'clip': torch.clamp})

except ModuleNotFoundError:
    pass


# TensorFlow
try:
    import tensorflow as tf

    bp.backend.set_buffer('tensorflow', {'clip': tf.clip_by_value})

except ModuleNotFoundError:
    pass


# Numba
try:
    import numba as nb

    @nb.njit
    def nb_clip(x, x_min, x_max):
        x = np.maximum(x, x_min)
        x = np.minimum(x, x_max)
        return x

    bp.backend.set_buffer('numba', {'clip': nb_clip})
    bp.backend.set_buffer('numba-parallel', {'clip': nb_clip})

except ModuleNotFoundError:
    pass

