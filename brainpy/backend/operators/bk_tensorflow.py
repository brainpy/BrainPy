# -*- coding: utf-8 -*-

"""
The TensorFlow with the version of xx is needed.
"""


from brainpy import errors

try:
    import tensorflow as tf
except ModuleNotFoundError:
    raise errors.PackageMissingError(errors.PackageMissingError(errors.backend_missing_msg.format(bk='tensorflow')))


reshape = tf.reshape
exp = tf.math.exp
sum = tf.math.reduce_sum
zeros = tf.zeros
eye = tf.eye
matmul = tf.matmul
arange = tf.range


def vstack(values):
    return tf.concat(values, axis=1)


def shape(x):
    if isinstance(x, (int, float)):
        return ()
    else:
        return x.shape()


def normal(loc, scale, size):
    return tf.random.normal(size, loc, scale)


def where(tensor, x, y):
    return tf.where(tensor, x, y)


unsqueeze = tf.expand_dims
squeeze = tf.squeeze

