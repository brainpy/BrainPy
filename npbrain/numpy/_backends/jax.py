# -*- coding: utf-8 -*-

from importlib import import_module

jax = None
key = None


def _check():
    global jax
    if jax is None:
        jax = import_module('jax')


def _get_subkey():
    _check()
    global key
    if key is None:
        key = jax.random.PRNGKey(0)

    key, subkey = jax.random.split(key)
    return subkey


def seed(seed=0):
    _check()
    global key
    key = jax.random.PRNGKey(seed)


def uniform(low=0.0, high=1.0, size=1):
    subkey = _get_subkey()
    return jax.random.uniform(subkey, size, minval=low, maxval=high)


def rand(*size):
    return uniform(low=0., high=1., size=size)


def randint(low, high=None, size=None):
    if high is None:
        low, high = 0, low
    subkey = _get_subkey()
    return jax.random.randint(subkey, size, minval=low, maxval=high)


def normal(mean = 0.0, stddev = 1.0, size = 1):
    subkey = _get_subkey()
    return jax.random.normal(subkey, size) * stddev + mean


def randn(*size):
    subkey = _get_subkey()
    return jax.random.normal(subkey, size)


def random(size=None):
    size = (1,) if size is None else size
    return uniform(low=0., high=1., size=size)


def random_sample():
    pass

def ranf():
    pass

def sample():
    pass

def choice():
    pass

def permutation():
    pass

def shuffle():
    pass


def beta(): pass


def binomial(): pass


def chisquare(): pass


def exponential(): pass


def f(): pass


def gamma(): pass


def geometric(): pass


def gumbel(): pass


def hypergeometric(): pass


def laplace(): pass


def logistic(): pass


def lognormal(): pass


def logseries(): pass


def multinomial(): pass


def negative_binomial(): pass



def pareto(): pass


def poisson(): pass


def power(): pass


def rayleigh(): pass


def standard_cauchy(): pass


def standard_exponential(): pass


def standard_gamma(): pass


def standard_normal(): pass


def standard_t(): pass


def triangular(): pass


def vonmises(): pass


def wald(): pass


def weibull(): pass


def zipf(): pass


