# -*- coding: utf-8 -*-

import numpy
import numpy.random

from brainpy import tools

try:
  import numba
except ModuleNotFoundError:
  numba = None

__all__ = [
  'seed', 'RandomState',
  'rand', 'randint', 'randn', 'random', 'random_sample',
  'ranf', 'sample', 'choice', 'permutation', 'shuffle',
  'beta', 'exponential', 'gamma', 'gumbel', 'laplace', 'logistic', 'normal', 'pareto',
  'poisson', 'standard_cauchy', 'standard_exponential', 'standard_gamma',
  'standard_normal', 'standard_t', 'uniform', 'bernoulli', 'truncated_normal',
  # 'binomial', 'chisquare', 'f', 'geometric', 'hypergeometric', 'lognormal',
  # 'logseries', 'multinomial', 'negative_binomial', 'power', 'rayleigh',
  # 'triangular', 'vonmises', 'wald', 'weibull',
]

# def RandomState(seed=None):
#   if seed:
#     if numba:
#       numba_seed(seed)
#     numpy.random.seed(seed)
#   return numpy.random

RandomState = numpy.random.RandomState
rand = numpy.random.rand
randint = numpy.random.randint
randn = numpy.random.randn
random = numpy.random.random
random_sample = numpy.random.random_sample
ranf = numpy.random.ranf
sample = numpy.random.sample
choice = numpy.random.choice
permutation = numpy.random.permutation
shuffle = numpy.random.shuffle
beta = numpy.random.beta
exponential = numpy.random.exponential
gamma = numpy.random.gamma
gumbel = numpy.random.gumbel
laplace = numpy.random.laplace
logistic = numpy.random.logistic
normal = numpy.random.normal
pareto = numpy.random.pareto
poisson = numpy.random.poisson
standard_cauchy = numpy.random.standard_cauchy
standard_exponential = numpy.random.standard_exponential
standard_gamma = numpy.random.standard_gamma
standard_normal = numpy.random.standard_normal
standard_t = numpy.random.standard_t
uniform = numpy.random.uniform


def bernoulli(p, size=None):
  return numpy.random.binomial(1, p=p, size=size)


def truncated_normal(lower, upper, size, scale=1.):
  raise NotImplementedError('Please use `brainpy.math.jax.random.truncated_normal()`')


@tools.numba_jit
def numba_seed(seed=None):
  numpy.random.seed(seed)


def seed(seed=None):
  if seed is not None:
    numpy.random.seed(seed)
    if numba:
      numba_seed(seed)
