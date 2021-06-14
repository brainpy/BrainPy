# -*- coding: utf-8 -*-

"""
The TensorFlow with the version of xx is needed.
"""

from brainpy import errors

try:
  import tensorflow as tf
except ModuleNotFoundError:
  raise errors.BackendNotInstalled('tensorflow')

from brainpy.backend.ops.more_unified_ops import tensorflow_

__all__ = [
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


# necessary ops for integrators

def normal(loc, scale, size):
  return tf.random.normal(size, loc, scale)


sum = tf.math.reduce_sum
exp = tf.math.exp


def shape(x):
  if not isinstance(x, tf.Tensor):
    return ()
  else:
    return x.shape()


# necessary ops for dynamics simulation

as_tensor = tf.convert_to_tensor
zeros = tf.zeros
ones = tf.ones
arange = tf.range
reshape = tf.reshape
concatenate = tf.concat


def vstack(values):
  return tf.concat(values, axis=1)


def where(tensor, x, y):
  return tf.where(tensor, x, y)


# necessary ops for dtypes

bool = tf.bool
int = tf.int32
int32 = tf.int32
int64 = tf.int64
float = tf.float32
float32 = tf.float32
float64 = tf.float64

if __name__ == '__main__':
  tensorflow_
