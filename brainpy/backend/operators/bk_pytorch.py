# -*- coding: utf-8 -*-

"""
The PyTorch with the version of xx is needed.
"""


import torch

as_tensor = torch.tensor
normal = torch.normal
reshape = torch.reshape
exp = torch.exp
sum = torch.sum
zeros = torch.zeros
ones = torch.ones
eye = torch.eye
outer = torch.outer
dot = torch.mm
vstack = torch.vstack
arange = torch.arange


def shape(x):
    if isinstance(x, (int, float)):
        return (1,)
    else:
        return x.size()

