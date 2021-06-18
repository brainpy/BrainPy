# -*- coding: utf-8 -*-

from jax.numpy import linalg

from brainpy.backend.math.jax.ndarray import _wrap

__all__ = [
  'cholesky', 'cond', 'det', 'eig', 'eigh', 'eigvals', 'eigvalsh', 'inv', 'svd',
  'lstsq', 'matrix_power', 'matrix_rank', 'norm', 'pinv', 'qr',
]

cholesky = _wrap(linalg.cholesky)
cond = _wrap(linalg.cond)
det = _wrap(linalg.det)
eig = _wrap(linalg.eig)
eigh = _wrap(linalg.eigh)
eigvals = _wrap(linalg.eigvals)
eigvalsh = _wrap(linalg.eigvalsh)
inv = _wrap(linalg.inv)
svd = _wrap(linalg.svd)
lstsq = _wrap(linalg.lstsq)
matrix_power = _wrap(linalg.matrix_power)
matrix_rank = _wrap(linalg.matrix_rank)
norm = _wrap(linalg.norm)
pinv = _wrap(linalg.pinv)
qr = _wrap(linalg.qr)
