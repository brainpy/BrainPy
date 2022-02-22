# -*- coding: utf-8 -*-

from jax.numpy import linalg

from brainpy.math.jaxarray import JaxArray
from brainpy.math.numpy_ops import _remove_jaxarray

__all__ = [
  'cholesky', 'cond', 'det', 'eig', 'eigh', 'eigvals', 'eigvalsh', 'inv', 'svd',
  'lstsq', 'matrix_power', 'matrix_rank', 'norm', 'pinv', 'qr', 'solve', 'slogdet',
  'tensorinv', 'tensorsolve', 'multi_dot'
]


def cholesky(a):
  a = _remove_jaxarray(a)
  return JaxArray(linalg.cholesky(a))


def cond(x, p=None):
  x = _remove_jaxarray(x)
  p = _remove_jaxarray(p)
  return linalg.cond(x, p=p)


def det(a):
  a = _remove_jaxarray(a)
  return JaxArray(linalg.det(a))


def eig(a):
  a = _remove_jaxarray(a)
  w, v = linalg.eig(a)
  return JaxArray(w), JaxArray(v)


def eigh(a, UPLO='L'):
  a = _remove_jaxarray(a)
  w, v = linalg.eigh(a, UPLO=UPLO)
  return JaxArray(w), JaxArray(v)


def eigvals(a):
  a = _remove_jaxarray(a)
  return JaxArray(linalg.eigvals(a))


def eigvalsh(a, UPLO='L'):
  a = _remove_jaxarray(a)
  return JaxArray(linalg.eigvalsh(a, UPLO=UPLO))


def inv(a):
  a = _remove_jaxarray(a)
  return JaxArray(linalg.inv(a))


def svd(a, full_matrices=True, compute_uv=True):
  a = _remove_jaxarray(a)
  u, s, vh = linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
  return JaxArray(u), JaxArray(s), JaxArray(vh)


def lstsq(a, b, rcond=None):
  a = _remove_jaxarray(a)
  b = _remove_jaxarray(b)
  x, resid, rank, s = linalg.lstsq(a, b, rcond=rcond)
  return JaxArray(x), JaxArray(resid), rank, JaxArray(s)


def matrix_power(a, n):
  a = _remove_jaxarray(a)
  return JaxArray(linalg.matrix_power(a, n))


def matrix_rank(M, tol=None):
  M = _remove_jaxarray(M)
  r = linalg.matrix_rank(M, tol=tol)
  return r if isinstance(r, int) else JaxArray(r)


def norm(x, ord=None, axis=None, keepdims=False):
  x = _remove_jaxarray(x)
  r = linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def pinv(a, rcond=None):
  a = _remove_jaxarray(a)
  if isinstance(rcond, JaxArray): rcond = rcond.value
  return JaxArray(linalg.pinv(a, rcond=rcond))


def qr(a, mode="reduced"):
  a = _remove_jaxarray(a)
  r = linalg.qr(a, mode=mode)
  if isinstance(r, tuple):
    return JaxArray(r[0]), JaxArray(r[1])
  else:
    return JaxArray(r)


def solve(a, b):
  a = _remove_jaxarray(a)
  b = _remove_jaxarray(b)
  return JaxArray(linalg.solve(a, b))


def slogdet(a):
  a = _remove_jaxarray(a)
  return JaxArray(linalg.slogdet(a))


def tensorinv(a, ind=2):
  a = _remove_jaxarray(a)
  return JaxArray(linalg.tensorinv(a, ind=ind))


def tensorsolve(a, b, axes=None):
  a = _remove_jaxarray(a)
  b = _remove_jaxarray(b)
  return JaxArray(linalg.tensorsolve(a, b, axes=axes))


def multi_dot(arrays):
  arrays = [_remove_jaxarray(a) for a in arrays]
  return JaxArray(linalg.multi_dot(arrays))
