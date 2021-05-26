# -*- coding: utf-8 -*-

from brainpy.backend import ops

import numpy as np

ops.set_buffer('numpy',
               clip=np.clip,
               unsqueeze=np.expand_dims,
               squeeze=np.squeeze,
               )
