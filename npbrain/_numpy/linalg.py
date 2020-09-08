# -*- coding: utf-8 -*-

import numpy.linalg

linear_algebra = [
    'cholesky', 'cond', 'det', 'eig', 'eigh', 'eigvals', 'eigvalsh', 'inv', 'svd',
    'lstsq', 'matrix_power', 'matrix_rank', 'norm', 'pinv', 'qr', 'slogdet', 'solve',
]

__all__ = []

for __ops in linear_algebra:
    __all__.append(getattr(numpy.linalg, __ops))


def _reload(backend):
    global_vars = globals()

    if backend == 'numpy':
        for __ops in linear_algebra:
            global_vars[__ops] = getattr(numpy.linalg, __ops)

    elif backend == 'numba':
        from ._backends import numba

        for __ops in linear_algebra:
            if hasattr(numba, __ops):
                global_vars[__ops] = getattr(numpy.linalg, __ops)
            else:
                global_vars[__ops] = getattr(numba, __ops)

    else:
        raise ValueError(f'Unknown backend device: {backend}')
