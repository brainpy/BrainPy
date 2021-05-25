# -*- coding: utf-8 -*-

import numpy as np

from brainpy.backend import ops

__all__ = []

ops.set_buffer('numpy',
               clip=np.clip,
               unsqueeze=np.expand_dims,
               squeeze=np.squeeze,
               )
