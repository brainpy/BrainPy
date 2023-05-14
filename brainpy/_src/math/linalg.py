# -*- coding: utf-8 -*-

from jax.numpy import linalg

from ._utils import _compatible_with_brainpy_array

__all__ = [
  'cholesky', 'cond', 'det', 'eig', 'eigh', 'eigvals', 'eigvalsh', 'inv', 'svd',
  'lstsq', 'matrix_power', 'matrix_rank', 'norm', 'pinv', 'qr', 'solve', 'slogdet',
  'tensorinv', 'tensorsolve', 'multi_dot'
]

cholesky = _compatible_with_brainpy_array(linalg.cholesky, module='linalg.')
cond = _compatible_with_brainpy_array(linalg.cond, module='linalg.')
det = _compatible_with_brainpy_array(linalg.det, module='linalg.')
eig = _compatible_with_brainpy_array(linalg.eig, module='linalg.')
eigh = _compatible_with_brainpy_array(linalg.eigh, module='linalg.')
eigvals = _compatible_with_brainpy_array(linalg.eigvals, module='linalg.')
eigvalsh = _compatible_with_brainpy_array(linalg.eigvalsh, module='linalg.')
inv = _compatible_with_brainpy_array(linalg.inv, module='linalg.')
svd = _compatible_with_brainpy_array(linalg.svd, module='linalg.')
lstsq = _compatible_with_brainpy_array(linalg.lstsq, module='linalg.')
matrix_power = _compatible_with_brainpy_array(linalg.matrix_power, module='linalg.')
matrix_rank = _compatible_with_brainpy_array(linalg.matrix_rank, module='linalg.')
norm = _compatible_with_brainpy_array(linalg.norm, module='linalg.')
pinv = _compatible_with_brainpy_array(linalg.pinv, module='linalg.')
qr = _compatible_with_brainpy_array(linalg.qr, module='linalg.')
solve = _compatible_with_brainpy_array(linalg.solve, module='linalg.')
slogdet = _compatible_with_brainpy_array(linalg.slogdet, module='linalg.')
tensorinv = _compatible_with_brainpy_array(linalg.tensorinv, module='linalg.')
tensorsolve = _compatible_with_brainpy_array(linalg.tensorsolve, module='linalg.')
multi_dot = _compatible_with_brainpy_array(linalg.multi_dot, module='linalg.')