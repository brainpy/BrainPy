# -*- coding: utf-8 -*-

from brainpy import errors

try:
  import jax
except ModuleNotFoundError:
  raise errors.BackendNotInstalled('jax')

from jax import numpy
from jax import random
from brainpy.backend.bpjax import more_ops

__all__ = [
  'set_seed',
  'normal',
  'exp',
  'sum',
  'shape',
  'as_tensor',
  'zeros',
  'ones',
  'arange',
  'concatenate',
  'where',
  'reshape',
  'bool',
  'int',
  'int32',
  'int64',
  'float',
  'float32',
  'float64'
]

key = random.PRNGKey(0)


def set_seed(seed):
  global key
  key = random.PRNGKey(seed)


# necessary ops for integrators

def normal(loc, scale, size):
  return loc + scale * random.normal(key, shape=size)


exp = numpy.exp
sum = numpy.sum
shape = numpy.shape

# necessary ops for dynamics simulation
as_tensor = numpy.asarray
zeros = numpy.zeros
ones = numpy.ones
arange = numpy.arange
concatenate = numpy.concatenate
where = numpy.where
reshape = numpy.reshape

# necessary ops for dtypes

bool = numpy.bool_
int = numpy.int_
int32 = numpy.int32
int64 = numpy.int64
float = numpy.float_
float32 = numpy.float32
float64 = numpy.float64

if __name__ == '__main__':
  more_ops
