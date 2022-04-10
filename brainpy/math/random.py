# -*- coding: utf-8 -*-

import numpy as np
from jax import numpy as jnp
from jax import random as jr
from jax.tree_util import register_pytree_node

from brainpy.math.jaxarray import JaxArray, Variable


__all__ = [
  'RandomState',

  'seed',

  'rand', 'randint', 'randn', 'random', 'random_sample', 'ranf', 'sample', 'choice', 'permutation', 'shuffle',
  'beta', 'exponential', 'gamma', 'gumbel', 'laplace', 'logistic', 'normal', 'pareto', 'poisson', 'standard_cauchy',
  'standard_exponential', 'standard_gamma', 'standard_normal', 'standard_t', 'uniform', 'truncated_normal', 'bernoulli',

  'lognormal',
]


def _size2shape(size):
  if size is None:
    return ()
  elif isinstance(size, int):
    return (size,)
  elif isinstance(size, (tuple, list)):
    return tuple(size)
  else:
    raise ValueError(f'Must be a list/tuple of int, but got {size}')


class RandomState(Variable):
  """RandomState that track the random generator state. """
  __slots__ = ()

  def __init__(self, seed=None):
    """RandomState constructor.

    Parameters
    ----------
    seed : int, jax.DeviceArray, Optional
      The initial seed of the random number generator.
    """
    if seed is None: seed = np.random.randint(0, 100000, 2, dtype=np.uint32)
    if isinstance(seed, int):
      key = jr.PRNGKey(seed)
    else:
      assert len(seed) == 2
      key = seed
    super(RandomState, self).__init__(key)

  # ------------------- #
  # seed and random key #
  # ------------------- #

  def seed(self, seed):
    """Sets a new random seed.

    Parameters
    ----------
    seed : int
      The new initial seed of the random number generator.
    """
    self.value = jr.PRNGKey(seed)

  def split_key(self):
    """Create a new seed from the current seed.
    """
    if not isinstance(self.value, jnp.ndarray):
      self.value = jnp.asarray(self.value)
    keys = jr.split(self.value, num=2)
    self.value = keys[0]
    return keys[1]

  def split_keys(self, n):
    """Create multiple seeds from the current seed. This is used
    internally by `pmap` and `vmap` to ensure that random numbers
    are different in parallel threads.

    Parameters
    ----------
    n : int
      The number of seeds to generate.
    """
    keys = jr.split(self.value, n + 1)
    self.value = keys[0]
    return keys[1:]

  # ---------------- #
  # random functions #
  # ---------------- #

  def rand(self, *dn):
    return JaxArray(jr.uniform(self.split_key(), shape=dn, minval=0., maxval=1.))

  def randint(self, low, high=None, size=None, dtype=int):
    return JaxArray(jr.randint(self.split_key(), shape=_size2shape(size),
                               minval=low, maxval=high, dtype=dtype))

  def randn(self, *dn):
    return JaxArray(jr.normal(self.split_key(), shape=dn))

  def random(self, size=None):
    return JaxArray(jr.uniform(self.split_key(), shape=_size2shape(size), minval=0., maxval=1.))

  def random_sample(self, size=None):
    return self.random(size=size)

  def randf(self, size=None):
    return self.random(size=size)

  def sample(self, size=None):
    return self.random(size=size)

  def choice(self, a, size=None, replace=True, p=None):
    a = a.value if isinstance(a, JaxArray) else a
    return JaxArray(jr.choice(self.split_key(), a=a, shape=_size2shape(size), replace=replace, p=p))

  def permutation(self, x):
    x = x.value if isinstance(x, JaxArray) else x
    return JaxArray(jr.permutation(self.split_key(), x))

  def shuffle(self, x, axis=0):
    assert isinstance(x, JaxArray), f'Must be a JaxArray, but got {type(x)}'
    x.value = jr.permutation(self.split_key(), x.value, axis=axis)

  def beta(self, a, b, size=None):
    a = a.value if isinstance(a, JaxArray) else a
    b = b.value if isinstance(b, JaxArray) else b
    return JaxArray(jr.beta(self.split_key(), a=a, b=b, shape=_size2shape(size)))

  def exponential(self, scale=1.0, size=None):
    assert scale == 1.
    return JaxArray(jr.exponential(self.split_key(), shape=_size2shape(size)))

  def gamma(self, shape, scale=1.0, size=None):
    assert scale == 1.
    return JaxArray(jr.gamma(self.split_key(), a=shape, shape=_size2shape(size)))

  def gumbel(self, loc=0.0, scale=1.0, size=None):
    assert loc == 0.
    assert scale == 1.
    return JaxArray(jr.gumbel(self.split_key(), shape=_size2shape(size)))

  def laplace(self, loc=0.0, scale=1.0, size=None):
    assert loc == 0.
    assert scale == 1.
    return JaxArray(jr.laplace(self.split_key(), shape=_size2shape(size)))

  def logistic(self, loc=0.0, scale=1.0, size=None):
    assert loc == 0.
    assert scale == 1.
    return JaxArray(jr.logistic(self.split_key(), shape=_size2shape(size)))

  def normal(self, loc=0.0, scale=1.0, size=None):
    return JaxArray(jr.normal(self.split_key(), shape=_size2shape(size)) * scale + loc)

  def pareto(self, a, size=None):
    return JaxArray(jr.pareto(self.split_key(), b=a, shape=_size2shape(size)))

  def poisson(self, lam=1.0, size=None):
    return JaxArray(jr.poisson(self.split_key(), lam=lam, shape=_size2shape(size)))

  def standard_cauchy(self, size=None):
    return JaxArray(jr.cauchy(self.split_key(), shape=_size2shape(size)))

  def standard_exponential(self, size=None):
    return JaxArray(jr.exponential(self.split_key(), shape=_size2shape(size)))

  def standard_gamma(self, shape, size=None):
    return JaxArray(jr.gamma(self.split_key(), a=shape, shape=_size2shape(size)))

  def standard_normal(self, size=None):
    return JaxArray(jr.normal(self.split_key(), shape=_size2shape(size)))

  def standard_t(self, df, size=None):
    return JaxArray(jr.t(self.split_key(), df=df, shape=_size2shape(size)))

  def uniform(self, low=0.0, high=1.0, size=None):
    return JaxArray(jr.uniform(self.split_key(), shape=_size2shape(size), minval=low, maxval=high))

  def truncated_normal(self, lower, upper, size, scale=1.):
    rands = jr.truncated_normal(self.split_key(),
                                lower=lower,
                                upper=upper,
                                shape=_size2shape(size))
    return JaxArray(rands * scale)

  def bernoulli(self, p, size=None):
    return JaxArray(jr.bernoulli(self.split_key(), p=p, shape=_size2shape(size)))

  def lognormallognormal(self, mean=0.0, sigma=1.0, size=None):
    samples = jr.normal(self.split_key(), shape=_size2shape(size))
    samples = samples * sigma + mean
    samples = jnp.exp(samples)
    return JaxArray(samples)



register_pytree_node(RandomState,
                     lambda t: ((t.value,), None),
                     lambda aux_data, flat_contents: RandomState(*flat_contents))

DEFAULT = RandomState(np.random.randint(0, 10000, size=2, dtype=np.uint32))


def seed(seed=None):
  global DEFAULT
  DEFAULT.seed(np.random.randint(0, 100000) if seed is None else seed)


def rand(*dn):
  return JaxArray(jr.uniform(DEFAULT.split_key(), shape=dn, minval=0., maxval=1.))


def randint(low, high=None, size=None, dtype=int):
  return JaxArray(jr.randint(DEFAULT.split_key(), shape=_size2shape(size),
                             minval=low, maxval=high, dtype=dtype))


def randn(*dn):
  return JaxArray(jr.normal(DEFAULT.split_key(), shape=dn))


def random(size=None):
  return JaxArray(jr.uniform(DEFAULT.split_key(), shape=_size2shape(size), minval=0., maxval=1.))


def random_sample(size=None):
  return JaxArray(jr.uniform(DEFAULT.split_key(), shape=_size2shape(size), minval=0., maxval=1.))


ranf = random_sample
sample = random_sample


def choice(a, size=None, replace=True, p=None):
  a = a.value if isinstance(a, JaxArray) else a
  return JaxArray(jr.choice(DEFAULT.split_key(), a=a, shape=_size2shape(size), replace=replace, p=p))


def permutation(x):
  x = x.value if isinstance(x, JaxArray) else x
  return JaxArray(jr.permutation(DEFAULT.split_key(), x))


def shuffle(x, axis=0):
  assert isinstance(x, JaxArray), f'Must be a JaxArray, but got {type(x)}'
  x.value = jr.permutation(DEFAULT.split_key(), x.value, axis=axis)


def beta(a, b, size=None):
  a = a.value if isinstance(a, JaxArray) else a
  b = b.value if isinstance(b, JaxArray) else b
  return JaxArray(jr.beta(DEFAULT.split_key(), a=a, b=b, shape=_size2shape(size)))


def exponential(scale=1.0, size=None):
  assert scale == 1.
  return JaxArray(jr.exponential(DEFAULT.split_key(), shape=_size2shape(size)))


def gamma(shape, scale=1.0, size=None):
  assert scale == 1.
  return JaxArray(jr.gamma(DEFAULT.split_key(), a=shape, shape=_size2shape(size)))


def gumbel(loc=0.0, scale=1.0, size=None):
  assert loc == 0.
  assert scale == 1.
  return JaxArray(jr.gumbel(DEFAULT.split_key(), shape=_size2shape(size)))


def laplace(loc=0.0, scale=1.0, size=None):
  assert loc == 0.
  assert scale == 1.
  return JaxArray(jr.laplace(DEFAULT.split_key(), shape=_size2shape(size)))


def logistic(loc=0.0, scale=1.0, size=None):
  assert loc == 0.
  assert scale == 1.
  return JaxArray(jr.logistic(DEFAULT.split_key(), shape=_size2shape(size)))


def normal(loc=0.0, scale=1.0, size=None):
  return JaxArray(jr.normal(DEFAULT.split_key(), shape=_size2shape(size)) * scale + loc)


def pareto(a, size=None):
  return JaxArray(jr.pareto(DEFAULT.split_key(), b=a, shape=_size2shape(size)))


def poisson(lam=1.0, size=None):
  return JaxArray(jr.poisson(DEFAULT.split_key(), lam=lam, shape=_size2shape(size)))


def standard_cauchy(size=None):
  return JaxArray(jr.cauchy(DEFAULT.split_key(), shape=_size2shape(size)))


def standard_exponential(size=None):
  return JaxArray(jr.exponential(DEFAULT.split_key(), shape=_size2shape(size)))


def standard_gamma(shape, size=None):
  return JaxArray(jr.gamma(DEFAULT.split_key(), a=shape, shape=_size2shape(size)))


def standard_normal(size=None):
  return JaxArray(jr.normal(DEFAULT.split_key(), shape=_size2shape(size)))


def standard_t(df, size=None):
  return JaxArray(jr.t(DEFAULT.split_key(), df=df, shape=_size2shape(size)))


def uniform(low=0.0, high=1.0, size=None):
  return JaxArray(jr.uniform(DEFAULT.split_key(), shape=_size2shape(size), minval=low, maxval=high))


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
  out : JaxArray
    A random array with the specified dtype and shape given by ``shape`` if
    ``shape`` is not None, or else by broadcasting ``lower`` and ``upper``.
    Returns values in the open interval ``(lower, upper)``.
  """
  rands = jr.truncated_normal(DEFAULT.split_key(),
                              lower=lower,
                              upper=upper,
                              shape=_size2shape(size))
  return JaxArray(rands * scale)


def bernoulli(p, size=None):
  """Sample Bernoulli random values with given shape and mean.

  Args:
    p: optional, a float or array of floats for the mean of the random
      variables. Must be broadcast-compatible with ``shape``. Default 0.5.
    size: optional, a tuple of nonnegative integers representing the result
      shape. Must be broadcast-compatible with ``p.shape``. The default (None)
      produces a result shape equal to ``p.shape``.

  Returns:
    A random array with boolean dtype and shape given by ``shape`` if ``shape``
    is not None, or else ``p.shape``.
  """
  return JaxArray(jr.bernoulli(DEFAULT.split_key(), p=p, shape=_size2shape(size)))


def lognormal(mean=0.0, sigma=1.0, size=None):
  samples = jr.normal(DEFAULT.split_key(), shape=_size2shape(size))
  samples = samples * sigma + mean
  samples = jnp.exp(samples)
  return JaxArray(samples)

