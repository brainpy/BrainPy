# -*- coding: utf-8 -*-

from importlib import import_module

import numpy.linalg

_operations = ['cholesky', 'cond', 'det', 'eig', 'eigh',
               'eigvals', 'eigvalsh', 'inv', 'lstsq',
               'matrix_power', 'matrix_rank', 'norm',
               'pinv', 'qr', 'slogdet', 'solve', 'svd', ]

__all__ = []

for __ops in _operations:
    __all__.append(getattr(numpy.linalg, __ops))


def _reload(backend):
    global_vars = globals()

    if backend in ['numpy', 'numba']:
        for __ops in _operations:
            global_vars[__ops] = getattr(numpy.linalg, __ops)

    elif backend == 'jax':
        # https://jax.readthedocs.io/en/latest/jax.random.html
        jnp = import_module('jax.numpy')

        for __ops in _operations:
            global_vars[__ops] = getattr(jnp.linalg, __ops)

    elif backend == 'torch':

        raise NotImplementedError

    elif backend == 'tensorflow':

        from ._backends import tensorflow

        for __ops in _operations:
            try:
                ops = getattr(tensorflow, __ops)
            except AttributeError:
                ops = None
            global_vars[__ops] = ops

    else:
        raise ValueError(f'Unknown backend device: {backend}')
