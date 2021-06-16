# -*- coding: utf-8 -*-

from brainpy.backend import ops

import torch

__all__ = []

ops.set_buffer('pytorch',
               clip=torch.clamp,
               unsqueeze=torch.unsqueeze,
               squeeze=torch.squeeze
               )
