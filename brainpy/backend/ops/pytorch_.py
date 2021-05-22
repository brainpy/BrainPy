# -*- coding: utf-8 -*-

"""
The PyTorch with the version of xx is needed.
"""

from brainpy import errors

try:
    import torch
except ModuleNotFoundError:
    raise errors.BackendNotInstalled('pytorch')

# necessary ops for integrators

normal = torch.normal
exp = torch.exp
sum = torch.sum


def shape(x):
    if isinstance(x, (int, float)):
        return ()
    else:
        return x.size()


# necessary ops for dynamics simulation

as_tensor = torch.as_tensor
zeros = torch.zeros
ones = torch.ones
arange = torch.arange
vstack = torch.vstack
reshape = torch.reshape


def where(tensor, x, y):
    if isinstance(x, (int, float)):
        x = torch.full_like(tensor, x)
    if isinstance(y, (int, float)):
        y = torch.full_like(tensor, y)
    return torch.where(tensor, x, y)


# necessary ops for dtypes

bool = torch.bool
int = torch.int
int32 = torch.int32
int64 = torch.int64
float = torch.float
float32 = torch.float32
float64 = torch.float64

