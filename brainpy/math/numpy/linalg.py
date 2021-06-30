# -*- coding: utf-8 -*-

import numpy


__all__ = [
  'cholesky', 'cond', 'det', 'eig', 'eigh', 'eigvals', 'eigvalsh', 'inv', 'svd',
  'lstsq', 'matrix_power', 'matrix_rank', 'norm', 'pinv', 'qr',
]

cholesky = numpy.linalg.cholesky
cond = numpy.linalg.cond
det = numpy.linalg.det
eig = numpy.linalg.eig
eigh = numpy.linalg.eigh
eigvals = numpy.linalg.eigvals
eigvalsh = numpy.linalg.eigvalsh
inv = numpy.linalg.inv
svd = numpy.linalg.svd
lstsq = numpy.linalg.lstsq
matrix_power = numpy.linalg.matrix_power
matrix_rank = numpy.linalg.matrix_rank
norm = numpy.linalg.norm
pinv = numpy.linalg.pinv
qr = numpy.linalg.qr

