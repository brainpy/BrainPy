# -*- coding: utf-8 -*-

import numba as nb
import numpy as np

from brainpy.backend import ops

__all__ = []


@nb.njit
def nb_clip(x, x_min, x_max):
  x = np.maximum(x, x_min)
  x = np.minimum(x, x_max)
  return x


operations = dict(
  clip=nb_clip,
  unsqueeze=np.expand_dims,
  squeeze=np.squeeze,
)

ops.set_buffer('numba', **operations)

ops.set_buffer('numba-parallel', **operations)
