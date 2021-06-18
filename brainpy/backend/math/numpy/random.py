# -*- coding: utf-8 -*-

import numpy

__all__ = [
  'seed',
  'rand', 'randint', 'randn', 'random', 'random_sample',
  'ranf', 'sample', 'choice', 'permutation', 'shuffle',
  'beta', 'binomial', 'chisquare', 'exponential', 'f',
  'gamma', 'geometric', 'gumbel', 'hypergeometric',
  'laplace', 'logistic', 'lognormal', 'logseries',
  'multinomial', 'negative_binomial', 'normal', 'pareto',
  'poisson', 'power', 'rayleigh', 'standard_cauchy',
  'standard_exponential', 'standard_gamma',
  'standard_normal', 'standard_t', 'triangular',
  'uniform', 'vonmises', 'wald', 'weibull',
]

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
binomial = numpy.random.binomial
chisquare = numpy.random.chisquare
exponential = numpy.random.exponential
f = numpy.random.f
gamma = numpy.random.gamma
geometric = numpy.random.geometric
gumbel = numpy.random.gumbel
hypergeometric = numpy.random.hypergeometric
laplace = numpy.random.laplace
logistic = numpy.random.logistic
lognormal = numpy.random.lognormal
logseries = numpy.random.logseries
multinomial = numpy.random.multinomial
negative_binomial = numpy.random.negative_binomial
normal = numpy.random.normal
_normal_like = lambda x: normal(size=numpy.shape(x))
pareto = numpy.random.pareto
poisson = numpy.random.poisson
power = numpy.random.power
rayleigh = numpy.random.rayleigh
standard_cauchy = numpy.random.standard_cauchy
standard_exponential = numpy.random.standard_exponential
standard_gamma = numpy.random.standard_gamma
standard_normal = numpy.random.standard_normal
standard_t = numpy.random.standard_t
triangular = numpy.random.triangular
uniform = numpy.random.uniform
vonmises = numpy.random.vonmises
wald = numpy.random.wald
weibull = numpy.random.weibull
