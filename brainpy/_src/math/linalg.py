# -*- coding: utf-8 -*-

from jax.numpy import linalg

from ._utils import _compatible_with_brainpy_array

__all__ = [
  'cholesky', 'cond', 'det', 'eig', 'eigh', 'eigvals', 'eigvalsh', 'inv', 'svd',
  'lstsq', 'matrix_power', 'matrix_rank', 'norm', 'pinv', 'qr', 'solve', 'slogdet',
  'tensorinv', 'tensorsolve', 'multi_dot'
]

cholesky = _compatible_with_brainpy_array(linalg.cholesky)
cond = _compatible_with_brainpy_array(linalg.cond)
det = _compatible_with_brainpy_array(linalg.det)
eig = _compatible_with_brainpy_array(linalg.eig)
eigh = _compatible_with_brainpy_array(linalg.eigh)
eigvals = _compatible_with_brainpy_array(linalg.eigvals)
eigvalsh = _compatible_with_brainpy_array(linalg.eigvalsh)
inv = _compatible_with_brainpy_array(linalg.inv)
svd = _compatible_with_brainpy_array(linalg.svd)
lstsq = _compatible_with_brainpy_array(linalg.lstsq)
matrix_power = _compatible_with_brainpy_array(linalg.matrix_power)
matrix_rank = _compatible_with_brainpy_array(linalg.matrix_rank)
norm = _compatible_with_brainpy_array(linalg.norm)
pinv = _compatible_with_brainpy_array(linalg.pinv)
qr = _compatible_with_brainpy_array(linalg.qr)
solve = _compatible_with_brainpy_array(linalg.solve)
slogdet = _compatible_with_brainpy_array(linalg.slogdet)
tensorinv = _compatible_with_brainpy_array(linalg.tensorinv)
tensorsolve = _compatible_with_brainpy_array(linalg.tensorsolve)
multi_dot = _compatible_with_brainpy_array(linalg.multi_dot)