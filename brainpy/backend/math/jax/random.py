# -*- coding: utf-8 -*-

import numpy.random

from brainpy.backend.math.jax.ndarray import _wrap


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
rand = _wrap(numpy.random.rand)
randint = _wrap(numpy.random.randint)
randn = _wrap(numpy.random.randn)
random = _wrap(numpy.random.random)
random_sample = _wrap(numpy.random.random_sample)
ranf = _wrap(numpy.random.ranf)
sample = _wrap(numpy.random.sample)
choice = _wrap(numpy.random.choice)
permutation = _wrap(numpy.random.permutation)
shuffle = _wrap(numpy.random.shuffle)
beta = _wrap(numpy.random.beta)
binomial = _wrap(numpy.random.binomial)
chisquare = _wrap(numpy.random.chisquare)
exponential = _wrap(numpy.random.exponential)
f = _wrap(numpy.random.f)
gamma = _wrap(numpy.random.gamma)
geometric = _wrap(numpy.random.geometric)
gumbel = _wrap(numpy.random.gumbel)
hypergeometric = _wrap(numpy.random.hypergeometric)
laplace = _wrap(numpy.random.laplace)
logistic = _wrap(numpy.random.logistic)
lognormal = _wrap(numpy.random.lognormal)
logseries = _wrap(numpy.random.logseries)
multinomial = _wrap(numpy.random.multinomial)
negative_binomial = _wrap(numpy.random.negative_binomial)
normal = _wrap(numpy.random.normal)
pareto = _wrap(numpy.random.pareto)
poisson = _wrap(numpy.random.poisson)
power = _wrap(numpy.random.power)
rayleigh = _wrap(numpy.random.rayleigh)
standard_cauchy = _wrap(numpy.random.standard_cauchy)
standard_exponential = _wrap(numpy.random.standard_exponential)
standard_gamma = _wrap(numpy.random.standard_gamma)
standard_normal = _wrap(numpy.random.standard_normal)
standard_t = _wrap(numpy.random.standard_t)
triangular = _wrap(numpy.random.triangular)
uniform = _wrap(numpy.random.uniform)
vonmises = _wrap(numpy.random.vonmises)
wald = _wrap(numpy.random.wald)
weibull = _wrap(numpy.random.weibull)

