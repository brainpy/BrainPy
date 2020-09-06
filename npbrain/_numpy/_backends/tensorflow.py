# -*- coding: utf-8 -*-

from importlib import import_module

from typing import Iterable

tf = None
tnp = None


def _check():
    global tf
    if tf is None:
        tf = import_module('tensorflow')
    global tnp
    if tnp is None:
        tnp = import_module('tensorflow.experimental.numpy')


###############################
# math operations
###############################


def fmod(x1, x2):
    _check()
    return tnp.remainder(x1, x2)


def trunc(x):
    _check()
    return tnp.where(x > 0., tnp.floor(x), -tnp.floor(-x))


def degrees(x):
    _check()
    return tnp.rad2deg(x)


def radians(x):
    _check()
    return tnp.deg2rad(x)


def invert(x):
    _check()
    return tnp.bitwise_not(x)


def fmin(x1, x2):
    _check()
    return tnp.where(x1 <= x2, x1, x2)


def fmax(x1, x2):
    _check()
    return tnp.where(x1 >= x2, x1, x2)


def column_stack(tup):
    _check()
    assert isinstance(tup, Iterable), 'Inputs must be a iterable object.'
    for t in tup:
        assert tnp.ndim(t) == 1, "Must be a 1D array."
    return tnp.vstack(tup).T


###############################
# linalg
###############################


def cholesky(tensor, name=None):
    _check()
    return tnp.asarray(tf.linalg.cholesky(tensor.data, name))


def det(tensor, name=None):
    _check()
    return tnp.asarray(tf.linalg.det(tensor.data, name))


def eig(tensor, name=None):
    _check()
    return tnp.asarray(tf.linalg.eig(tensor.data, name))


def eigh(tensor, name=None):
    _check()
    return tnp.asarray(tf.linalg.eigh(tensor.data, name))


def eigvals(tensor, name=None):
    _check()
    return tnp.asarray(tf.linalg.eigvals(tensor.data, name))


def eigvalsh(tensor, name=None):
    _check()
    return tnp.asarray(tf.linalg.eigvalsh(tensor.data, name))


def inv(tensor, adjoint=False, name=None):
    _check()
    return tnp.asarray(tf.linalg.inv(tensor.data, adjoint, name))


def lstsq(matrix, rhs, l2_regularizer=0.0, fast=True, name=None):
    _check()
    return tnp.asarray(tf.linalg.lstsq(matrix.data, rhs.data, l2_regularizer, fast, name))


def matrix_rank(a, tol=None, validate_args=False, name=None):
    _check()
    return tnp.asarray(tf.linalg.matrix_rank(a.data, tol, validate_args, name))


def norm(tensor, ord='euclidean', axis=None, keepdims=None, name=None):
    _check()
    return tnp.asarray(tf.linalg.norm(tensor.data, ord, axis, keepdims, name))


def pinv(a, rcond=None, validate_args=False, name=None):
    _check()
    return tnp.asarray(tf.linalg.pinv(a.data, rcond, validate_args, name))


def qr(input, full_matrices=False, name=None):
    _check()
    return tnp.asarray(tf.linalg.qr(input.data, full_matrices, name))


def slogdet(input, name=None):
    _check()
    return tnp.asarray(tf.linalg.slogdet(input.data, name))


def solve(matrix, rhs, adjoint=False, name=None):
    _check()
    return tnp.asarray(tf.linalg.solve(matrix.data, rhs.data, adjoint, name))


def svd(tensor, full_matrices=False, compute_uv=True, name=None):
    _check()
    return tnp.asarray(tf.linalg.svd(tensor.data, full_matrices, compute_uv, name))

###############################
# signal
###############################
