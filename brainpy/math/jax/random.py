# -*- coding: utf-8 -*-


import numpy as np
from jax import random as jr

from brainpy.math.jax.ndarray import ndarray

__all__ = [
  'seed', 'rand', 'randint', 'randn', 'random', 'random_sample',
  'ranf', 'sample', 'choice', 'permutation', 'shuffle',
  'beta', 'exponential', 'gamma', 'gumbel',
  'laplace', 'logistic', 'normal', 'pareto',
  'poisson', 'standard_cauchy', 'standard_exponential',
  'standard_gamma', 'standard_normal', 'standard_t',
  'uniform', 'truncated_normal',
]


def _copy_doc(source_f):
  def copy(target_f):
    target_f.__doc__ = source_f.__doc__
    return target_f

  return copy


def _size2shape(size):
  if size is None:
    return ()
  elif isinstance(size, int):
    return (size,)
  elif isinstance(size, (tuple, list)):
    return tuple(size)
  else:
    raise ValueError(f'Must be a list/tuple of int, but got {size}')


class RandomState(object):
  """RandomState are variables that track the
  random generator state. They are meant to be used internally.
  Currently only the random.Generator module uses them."""

  @property
  def key(self):
    return self._key

  @key.setter
  def key(self, key):
    self._key = key

  def __init__(self, seed=0):
    """RandomState constructor.

    Args:
        seed: the initial seed of the random number generator.
    """
    self._key = jr.PRNGKey(seed)

  def seed(self, seed):
    """Sets a new random seed.

    Args:
        seed: the new initial seed of the random number generator.
    """
    self._key = jr.PRNGKey(seed)

  def split(self):
    """Create multiple seeds from the current seed. This is used internally by Parallel and Vectorize to ensure
    that random numbers are different in parallel threads.

    Args:
        n: the number of seeds to generate.
    """
    keys = jr.split(self.key, 2)
    self._key = keys[0]
    return keys[1]

  def splits(self, n):
    """Create multiple seeds from the current seed. This is used internally by Parallel and Vectorize to ensure
    that random numbers are different in parallel threads.

    Args:
        n: the number of seeds to generate.
    """
    keys = jr.split(self.key, n + 1)
    self._key = keys[0]
    return keys[1:]


_ST = RandomState(np.random.randint(100, 100000))


@_copy_doc(np.random.seed)
def seed(seed=None):
  global _ST
  _ST.seed(np.random.randint(100, 100000) if seed is None else seed)


@_copy_doc(np.random.rand)
def rand(*dn):
  return ndarray(jr.uniform(_ST.split(), shape=dn, minval=0., maxval=1.))


@_copy_doc(np.random.randint)
def randint(low, high=None, size=None, dtype=int):
  return ndarray(jr.randint(_ST.split(), shape=_size2shape(size),
                            minval=low, maxval=high, dtype=dtype))


@_copy_doc(np.random.randn)
def randn(*dn):
  return ndarray(jr.normal(_ST.split(), shape=dn))


@_copy_doc(np.random.random)
def random(size=None):
  return ndarray(jr.uniform(_ST.split(), shape=_size2shape(size), minval=0., maxval=1.))


@_copy_doc(np.random.random_sample)
def random_sample(size=None):
  return ndarray(jr.uniform(_ST.split(), shape=_size2shape(size), minval=0., maxval=1.))


ranf = random_sample
sample = random_sample


@_copy_doc(np.random.choice)
def choice(a, size=None, replace=True, p=None):
  a = a.value if isinstance(a, ndarray) else a
  return ndarray(jr.choice(_ST.split(), a=a, shape=_size2shape(size), replace=replace, p=p))


@_copy_doc(np.random.permutation)
def permutation(x):
  x = x.value if isinstance(x, ndarray) else x
  return ndarray(jr.permutation(_ST.split(), x))


@_copy_doc(np.random.shuffle)
def shuffle(x, axis=0):
  x = x.value if isinstance(x, ndarray) else x
  return ndarray(jr.shuffle(_ST.split(), x, axis=axis))


@_copy_doc(np.random.beta)
def beta(a, b, size=None):
  a = a.value if isinstance(a, ndarray) else a
  b = b.value if isinstance(b, ndarray) else b
  return ndarray(jr.beta(_ST.split(), a=a, b=b, shape=_size2shape(size)))


@_copy_doc(np.random.exponential)
def exponential(scale=1.0, size=None):
  assert scale == 1.
  return ndarray(jr.exponential(_ST.split(), shape=_size2shape(size)))


@_copy_doc(np.random.gamma)
def gamma(shape, scale=1.0, size=None):
  assert scale == 1.
  return ndarray(jr.gamma(_ST.split(), a=shape, shape=_size2shape(size)))


@_copy_doc(np.random.gumbel)
def gumbel(loc=0.0, scale=1.0, size=None):
  assert loc == 0.
  assert scale == 1.
  return ndarray(jr.gumbel(_ST.split(), shape=_size2shape(size)))


@_copy_doc(np.random.laplace)
def laplace(loc=0.0, scale=1.0, size=None):
  assert loc == 0.
  assert scale == 1.
  return ndarray(jr.laplace(_ST.split(), shape=_size2shape(size)))


@_copy_doc(np.random.logistic)
def logistic(loc=0.0, scale=1.0, size=None):
  assert loc == 0.
  assert scale == 1.
  return ndarray(jr.logistic(_ST.split(), shape=_size2shape(size)))


@_copy_doc(np.random.normal)
def normal(loc=0.0, scale=1.0, size=None):
  return ndarray(jr.normal(_ST.split(), shape=_size2shape(size)) * scale + loc)


@_copy_doc(np.random.pareto)
def pareto(a, size=None):
  return ndarray(jr.pareto(_ST.split(), b=a, shape=_size2shape(size)))


@_copy_doc(np.random.poisson)
def poisson(lam=1.0, size=None):
  return ndarray(jr.poisson(_ST.split(), lam=lam, shape=_size2shape(size)))


@_copy_doc(np.random.standard_cauchy)
def standard_cauchy(size=None):
  return ndarray(jr.cauchy(_ST.split(), shape=_size2shape(size)))


@_copy_doc(np.random.standard_exponential)
def standard_exponential(size=None):
  return ndarray(jr.exponential(_ST.split(), shape=_size2shape(size)))


@_copy_doc(np.random.standard_gamma)
def standard_gamma(shape, size=None):
  return ndarray(jr.gamma(_ST.split(), a=shape, shape=_size2shape(size)))


@_copy_doc(np.random.standard_normal)
def standard_normal(size=None):
  return ndarray(jr.normal(_ST.split(), shape=_size2shape(size)))


@_copy_doc(np.random.standard_t)
def standard_t(df, size=None):
  return ndarray(jr.t(_ST.split(), df=df, shape=_size2shape(size)))


@_copy_doc(np.random.uniform)
def uniform(low=0.0, high=1.0, size=None):
  return ndarray(jr.uniform(_ST.split(), shape=_size2shape(size), minval=low, maxval=high))


def truncated_normal(lower, upper, size, scale=1.):
  """Sample truncated standard normal random values with given shape and dtype.

  Parameters
  ----------
  lower : float, ndarray
    A float or array of floats representing the lower bound for
    truncation. Must be broadcast-compatible with ``upper``.
  upper : float, ndarray
    A float or array of floats representing the  upper bound for
    truncation. Must be broadcast-compatible with ``lower``.
  size : optional, list of int, tuple of int
    A tuple of nonnegative integers specifying the result
    shape. Must be broadcast-compatible with ``lower`` and ``upper``. The
    default (None) produces a result shape by broadcasting ``lower`` and
    ``upper``.
  scale : float, ndarray
    Standard deviation (spread or "width") of the distribution. Must be
    non-negative.

  Returns
  -------
  out : ndarray
    A random array with the specified dtype and shape given by ``shape`` if
    ``shape`` is not None, or else by broadcasting ``lower`` and ``upper``.
    Returns values in the open interval ``(lower, upper)``.
  """
  return ndarray(jr.truncated_normal(_ST.split(), lower=lower, upper=upper, shape=size) * scale)

