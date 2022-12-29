# -*- coding: utf-8 -*-

from jax.numpy import linalg

from brainpy.math.ndarray import Array
from brainpy.math.numpy_ops import _as_jax_array_

__all__ = [
  'cholesky', 'cond', 'det', 'eig', 'eigh', 'eigvals', 'eigvalsh', 'inv', 'svd',
  'lstsq', 'matrix_power', 'matrix_rank', 'norm', 'pinv', 'qr', 'solve', 'slogdet',
  'tensorinv', 'tensorsolve', 'multi_dot'
]


def cholesky(a):
  a = _as_jax_array_(a)
  return linalg.cholesky(a)


def cond(x, p=None):
  x = _as_jax_array_(x)
  p = _as_jax_array_(p)
  return linalg.cond(x, p=p)


def det(a):
  a = _as_jax_array_(a)
  return linalg.det(a)


def eig(a):
  a = _as_jax_array_(a)
  w, v = linalg.eig(a)
  return w, v


def eigh(a, UPLO='L'):
  a = _as_jax_array_(a)
  w, v = linalg.eigh(a, UPLO=UPLO)
  return w, v


def eigvals(a):
  a = _as_jax_array_(a)
  return linalg.eigvals(a)


def eigvalsh(a, UPLO='L'):
  a = _as_jax_array_(a)
  return linalg.eigvalsh(a, UPLO=UPLO)


def inv(a):
  a = _as_jax_array_(a)
  return linalg.inv(a)


def svd(a, full_matrices=True, compute_uv=True):
  a = _as_jax_array_(a)
  return linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)


def lstsq(a, b, rcond=None):
  a = _as_jax_array_(a)
  b = _as_jax_array_(b)
  return linalg.lstsq(a, b, rcond=rcond)


def matrix_power(a, n):
  a = _as_jax_array_(a)
  return linalg.matrix_power(a, n)


def matrix_rank(M, tol=None):
  M = _as_jax_array_(M)
  return linalg.matrix_rank(M, tol=tol)


def norm(x, ord=None, axis=None, keepdims=False):
  x = _as_jax_array_(x)
  return linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)


def pinv(a, rcond=None):
  a = _as_jax_array_(a)
  rcond = _as_jax_array_(rcond)
  return linalg.pinv(a, rcond=rcond)


def qr(a, mode="reduced"):
  a = _as_jax_array_(a)
  return linalg.qr(a, mode=mode)


def solve(a, b):
  a = _as_jax_array_(a)
  b = _as_jax_array_(b)
  return linalg.solve(a, b)


def slogdet(a):
  a = _as_jax_array_(a)
  return linalg.slogdet(a)


def tensorinv(a, ind=2):
  a = _as_jax_array_(a)
  return linalg.tensorinv(a, ind=ind)


def tensorsolve(a, b, axes=None):
  a = _as_jax_array_(a)
  b = _as_jax_array_(b)
  return linalg.tensorsolve(a, b, axes=axes)


def multi_dot(arrays, *, precision=None):
  arrays = [_as_jax_array_(a) for a in arrays]
  return linalg.multi_dot(arrays, precision=precision)
