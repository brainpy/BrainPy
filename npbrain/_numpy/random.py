# -*- coding: utf-8 -*-

from importlib import import_module

import numpy.random

random_distribution = ['uniform', 'seed', 'rand', 'randint', 'randn', 'random']

__all__ = []

for __ops in random_distribution:
    __all__.append(getattr(numpy.random, __ops))


def _reload(backend):
    global_vars = globals()

    if backend in ['numpy', 'numba']:
        for __ops in random_distribution:
            global_vars[__ops] = getattr(numpy.random, __ops)

    elif backend == 'jax':
        # https://jax.readthedocs.io/en/latest/jax.random.html
        from npbrain._numpy._backends import jax

        for __ops in random_distribution:
            global_vars[__ops] = getattr(jax, __ops)

    elif backend == 'torch':
        from npbrain._numpy._backends import pytorch

        for __ops in random_distribution:
            global_vars[__ops] = getattr(pytorch, __ops)

    elif backend == 'tensorflow':
        # https://www.tensorflow.org/api_docs/python/tf/experimental/numpy/random
        trandom = import_module('tensorflow.experimental.numpy.random')

        for __ops in random_distribution:
            global_vars[__ops] = getattr(trandom, __ops)

    else:
        raise ValueError(f'Unknown backend device: {backend}')
