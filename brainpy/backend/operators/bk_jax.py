# -*- coding: utf-8 -*-

from brainpy import errors

try:
    import jax
except ModuleNotFoundError:
    raise errors.PackageMissingError(errors.PackageMissingError(errors.backend_missing_msg.format(bk='jax')))

from jax import numpy
from jax import random

key = random.PRNGKey(0)


def set_seed(seed):
    global key
    key = random.PRNGKey(seed)


# necessary ops for integrators

def normal(loc, scale, size):
    return loc + scale * random.normal(key, shape=size)


exp = numpy.exp
sum = numpy.sum
matmul = numpy.matmul
shape = numpy.shape


# necessary ops for dynamics simulation
as_tensor = numpy.asarray
zeros = numpy.zeros
ones = numpy.ones
arange = numpy.arange
vstack = numpy.vstack
where = numpy.where
unsqueeze = numpy.expand_dims
squeeze = numpy.squeeze
