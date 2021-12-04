# -*- coding: utf-8 -*-

from jax.numpy import linalg

from brainpy.math.jaxarray import JaxArray

__all__ = [
  'cholesky', 'cond', 'det', 'eig', 'eigh', 'eigvals', 'eigvalsh', 'inv', 'svd',
  'lstsq', 'matrix_power', 'matrix_rank', 'norm', 'pinv', 'qr',
]


def cholesky(a):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(linalg.cholesky(a))


def cond(x, p=None):
  if isinstance(x, JaxArray): x = x.value
  if isinstance(p, JaxArray): p = p.value
  return linalg.cond(x, p=p)


def det(a):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(linalg.det(a))


def eig(a):
  if isinstance(a, JaxArray): a = a.value
  w, v = linalg.eig(a)
  return JaxArray(w), JaxArray(v)


def eigh(a, UPLO='L'):
  if isinstance(a, JaxArray): a = a.value
  w, v = linalg.eigh(a, UPLO=UPLO)
  return JaxArray(w), JaxArray(v)


def eigvals(a):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(linalg.eigvals(a))


def eigvalsh(a, UPLO='L'):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(linalg.eigvalsh(a, UPLO=UPLO))


def inv(a):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(linalg.inv(a))


def svd(a, full_matrices=True, compute_uv=True):
  if isinstance(a, JaxArray): a = a.value
  u, s, vh = linalg.svd(a, full_matrices=full_matrices, compute_uv=compute_uv)
  return JaxArray(u), JaxArray(s), JaxArray(vh)


def lstsq(a, b, rcond=None):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(b, JaxArray): b = b.value
  x, resid, rank, s = linalg.lstsq(a, b, rcond=rcond)
  return JaxArray(x), JaxArray(resid), rank, JaxArray(s)


def matrix_power(a, n):
  if isinstance(a, JaxArray): a = a.value
  return JaxArray(linalg.matrix_power(a, n))


def matrix_rank(M, tol=None):
  if isinstance(M, JaxArray): M = M.value
  r = linalg.matrix_rank(M, tol=tol)
  return r if isinstance(r, int) else JaxArray(r)


def norm(x, ord=None, axis=None, keepdims=False):
  if isinstance(x, JaxArray): x = x.value
  r = linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
  return r if axis is None else JaxArray(r)


def pinv(a, rcond=None):
  if isinstance(a, JaxArray): a = a.value
  if isinstance(rcond, JaxArray): rcond = rcond.value
  return JaxArray(linalg.pinv(a, rcond=rcond))


def qr(a, mode="reduced"):
  if isinstance(a, JaxArray): a = a.value
  r = linalg.qr(a, mode=mode)
  if isinstance(r, tuple):
    return JaxArray(r[0]), JaxArray(r[1])
  else:
    return JaxArray(r)
