# -*- coding: utf-8 -*-

import numpy.random

random_functions = [
    # Initialization
    # ----------------
    'seed',

    # Simple random data
    # -------------------
    'rand', 'randint', 'randn', 'random', 'random_sample', 'ranf', 'sample',

    # Permutations
    # -------------------
    'choice', 'permutation', 'shuffle',

    # Distributions
    # ---------------
    'beta', 'binomial', 'chisquare', 'exponential', 'f', 'gamma', 'geometric', 'gumbel',
    'hypergeometric', 'laplace', 'logistic', 'lognormal', 'logseries', 'multinomial',
    'negative_binomial', 'normal', 'pareto', 'poisson', 'power', 'rayleigh', 'standard_cauchy',
    'standard_exponential', 'standard_gamma', 'standard_normal', 'standard_t', 'triangular',
    'uniform', 'vonmises', 'wald', 'weibull', 'zipf',
]

__all__ = []

for __ops in random_functions:
    __all__.append(getattr(numpy.random, __ops))


def _reload(backend):
    global_vars = globals()

    if backend == 'numpy':
        for __ops in random_functions:
            global_vars[__ops] = getattr(numpy.random, __ops)

    if backend == 'numba':
        from ._backends import numba

        for __ops in random_functions:
            if hasattr(numba, __ops):
                global_vars[__ops] = getattr(numpy.random, __ops)
            else:
                global_vars[__ops] = getattr(numba, __ops)

    else:
        raise ValueError(f'Unknown backend device: {backend}')
