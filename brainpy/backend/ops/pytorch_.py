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
matmul = torch.matmul


def shape(x):
    if isinstance(x, (int, float)):
        return ()
    else:
        return x.size()


# necessary ops for dynamics simulation

as_tensor = torch.tensor
zeros = torch.zeros
ones = torch.ones
arange = torch.arange
vstack = torch.vstack


def where(tensor, x, y):
    if isinstance(x, (int, float)):
        x = torch.full_like(tensor, x)
    if isinstance(y, (int, float)):
        y = torch.full_like(tensor, y)
    return torch.where(tensor, x, y)


unsqueeze = torch.unsqueeze
squeeze = torch.squeeze
