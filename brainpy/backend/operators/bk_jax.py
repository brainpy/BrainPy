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


def normal(loc, scale, size):
    return loc + scale * random.normal(key, shape=size)


reshape = numpy.reshape
exp = numpy.exp
sum = numpy.sum
zeros = numpy.zeros
eye = numpy.eye
matmul = numpy.matmul
vstack = numpy.vstack
arange = numpy.arange


def shape(x):
    size = numpy.shape(x)
    if len(size) == 0:
        return (1,)
    else:
        return size
