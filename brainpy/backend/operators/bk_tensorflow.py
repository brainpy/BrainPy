# -*- coding: utf-8 -*-

"""
The TensorFlow with the version of xx is needed.
"""

from brainpy import errors

try:
    import tensorflow as tf
except ModuleNotFoundError:
    raise errors.PackageMissingError(errors.backend_missing_msg.format(bk='tensorflow'))


# necessary ops for integrators

def normal(loc, scale, size):
    return tf.random.normal(size, loc, scale)


sum = tf.math.reduce_sum
exp = tf.math.exp
matmul = tf.matmul


def shape(x):
    if isinstance(x, (int, float)):
        return ()
    else:
        return x.shape()


# necessary ops for dynamics simulation

as_tensor = tf.convert_to_tensor
zeros = tf.zeros
ones = tf.ones
arange = tf.range


def vstack(values):
    return tf.concat(values, axis=1)


def where(tensor, x, y):
    return tf.where(tensor, x, y)


unsqueeze = tf.expand_dims
squeeze = tf.squeeze
