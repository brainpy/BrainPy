# -*- coding: utf-8 -*-

"""
The PyTorch with the version of xx is needed.
"""

from brainpy import errors

try:
    import torch
except ModuleNotFoundError:
    raise errors.PackageMissingError(errors.PackageMissingError(errors.backend_missing_msg.format(bk='pytorch')))

as_tensor = torch.tensor
normal = torch.normal
reshape = torch.reshape
exp = torch.exp
sum = torch.sum
zeros = torch.zeros
ones = torch.ones
eye = torch.eye
matmul = torch.matmul
vstack = torch.vstack
arange = torch.arange


def shape(x):
    if isinstance(x, (int, float)):
        return ()
    else:
        return x.size()


def where(tensor, x, y):
    if isinstance(x, (int, float)):
        x = torch.full_like(tensor, x)
    if isinstance(y, (int, float)):
        y = torch.full_like(tensor, y)
    return torch.where(tensor, x, y)


unsqueeze = torch.unsqueeze
squeeze = torch.squeeze

