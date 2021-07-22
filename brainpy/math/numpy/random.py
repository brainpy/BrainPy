# -*- coding: utf-8 -*-

import numpy

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

RandomState = numpy.random.RandomState
seed = numpy.random.seed
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


def truncated_normal():
  raise NotImplementedError
