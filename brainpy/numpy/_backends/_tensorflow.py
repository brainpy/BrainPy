# -*- coding: utf-8 -*-

"""
API comes from:

https://www.tensorflow.org/api_docs/python/tf/experimental/numpy


"""

from importlib import import_module
from typing import Iterable

tf = import_module('tensorflow')
tnp = import_module('tensorflow.experimental.numpy')


# ----------------
# Normal function
# ----------------


def fmod(x1, x2):
    return tnp.remainder(x1, x2)


def trunc(x):
    return tnp.where(x > 0., tnp.floor(x), -tnp.floor(-x))


def degrees(x):
    return tnp.rad2deg(x)


def radians(x):
    return tnp.deg2rad(x)


def invert(x):
    return tnp.bitwise_not(x)


def fmin(x1, x2):
    return tnp.where(x1 <= x2, x1, x2)


def fmax(x1, x2):
    return tnp.where(x1 >= x2, x1, x2)


def column_stack(tup):
    assert isinstance(tup, Iterable), 'Inputs must be a iterable object.'
    for t in tup:
        assert tnp.ndim(t) == 1, "Must be a 1D array."
    return tnp.vstack(tup).T


def modf(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def round_(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def rint(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def nancumprod(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def nancumsum(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def ediff1d(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def trapz(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def copysign(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def ldexp(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def frexp(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def spacing(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def convolve(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def interp(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def left_shift(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def right_shift(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def unique(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def delete(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def partition(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def argwhere(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def flatnonzero(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def searchsorted(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def extract(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def tril_indices(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def tril_indices_from(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def triu_indices(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def triu_indices_from(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def nditer(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def ndenumerate(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def ndindex(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def nanmin(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def nanmax(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def percentile(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def nanpercentile(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def quantile(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def nanquantile(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def median(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def nanmedian(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def nanstd(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def nanvar(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def corrcoef(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def correlate(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def cov(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def histogram(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def bincount(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def digitize(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def bartlett(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def blackman(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def hamming(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def hanning(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def kaiser(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def dtype(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def MachAr(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


# -----------------
# Random function
# -----------------


def standard_normal(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def poisson(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def random_sample(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def ranf(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def sample(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def choice(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def permutation(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def shuffle(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def beta(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def binomial(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def chisquare(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def exponential(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def f(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def gamma(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def geometric(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def gumbel(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def hypergeometric(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def laplace(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def logistic(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def lognormal(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def logseries(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def multinomial(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def negative_binomial(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def normal(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def pareto(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def power(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def rayleigh(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def standard_cauchy(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def standard_exponential(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def standard_gamma(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def standard_t(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def triangular(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def vonmises(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def wald(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def weibull(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def zipf(**kwargs):
    raise NotImplementedError("Don't support in TensorFlow backend.")


def _normal_like(x):
    return tnp.random.randn(*tnp.shape(x))


# ----------------
# Linear algebra
# ----------------


def cond():
    pass


def cholesky(tensor, name=None):
    return tnp.array(tf.linalg.cholesky(tensor.data, name))


def det(tensor, name=None):
    return tnp.array(tf.linalg.det(tensor.data, name))


def eig(tensor, name=None):
    return tnp.array(tf.linalg.eig(tensor.data, name))


def eigh(tensor, name=None):
    return tnp.array(tf.linalg.eigh(tensor.data, name))


def eigvals(tensor, name=None):
    return tnp.array(tf.linalg.eigvals(tensor.data, name))


def eigvalsh(tensor, name=None):
    return tnp.array(tf.linalg.eigvalsh(tensor.data, name))


def inv(tensor, adjoint=False, name=None):
    return tnp.array(tf.linalg.inv(tensor.data, adjoint, name))


def lstsq(matrix, rhs, l2_regularizer=0.0, fast=True, name=None):
    return tnp.array(tf.linalg.lstsq(matrix.data, rhs.data, l2_regularizer, fast, name))


def matrix_rank(a, tol=None, validate_args=False, name=None):
    return tnp.array(tf.linalg.matrix_rank(a.data, tol, validate_args, name))


def norm(tensor, ord='euclidean', axis=None, keepdims=None, name=None):
    return tnp.array(tf.linalg.norm(tensor.data, ord, axis, keepdims, name))


def pinv(a, rcond=None, validate_args=False, name=None):
    return tnp.array(tf.linalg.pinv(a.data, rcond, validate_args, name))


def qr(input, full_matrices=False, name=None):
    return tnp.array(tf.linalg.qr(input.data, full_matrices, name))


def slogdet(input, name=None):
    return tnp.array(tf.linalg.slogdet(input.data, name))


def solve(matrix, rhs, adjoint=False, name=None):
    return tnp.array(tf.linalg.solve(matrix.data, rhs.data, adjoint, name))


def svd(tensor, full_matrices=False, compute_uv=True, name=None):
    return tnp.array(tf.linalg.svd(tensor.data, full_matrices, compute_uv, name))


def matrix_power():
    pass
