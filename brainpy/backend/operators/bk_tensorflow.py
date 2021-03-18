# -*- coding: utf-8 -*-


"""
The TensorFlow with the version of xx is needed.
"""

import tensorflow as tf

reshape = tf.reshape
exp = tf.math.exp
sum = tf.math.reduce_sum
zeros = tf.zeros
eye = tf.eye
dot = tf.matmul
arange = tf.range


def outer(A, B):
    return tf.tensordot(A, B, axes=0)


def vstack(values):
    return tf.concat(values, axis=1)


def shape(x):
    if isinstance(x, (int, float)):
        return (1,)
    else:
        return x.shape()


def normal(loc, scale, size):
    return tf.random.normal(size, loc, scale)
