# -*- coding: utf-8 -*-
import jax.experimental.host_callback
import numpy as np
import numpy.random
from jax import numpy as jnp
from jax import random as jr
from jax.tree_util import register_pytree_node

from brainpy.math.jaxarray import JaxArray, Variable

from .utils import wraps
from jax.experimental.host_callback import call as hcb_call


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


def _check_shape(name, shape, *param_shapes):
  for s in param_shapes:
    if s != shape:
      msg = ("{} parameter shapes must be broadcast-compatible with shape "
             "argument, and the result of broadcasting the shapes must equal "
             "the shape argument, but got result {} for shape argument {}.")
      raise ValueError(msg.format(name, s, shape))


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
      self._value = jnp.asarray(self.value)
    keys = jr.split(self.value, num=2)
    self._value = keys[0]
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
    self._value = keys[0]
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


@wraps(np.random.seed)
def seed(seed=None):
  global DEFAULT
  DEFAULT.seed(np.random.randint(0, 100000) if seed is None else seed)


@wraps(np.random.rand)
def rand(*dn):
  return JaxArray(jr.uniform(DEFAULT.split_key(), shape=dn, minval=0., maxval=1.))


@wraps(np.random.randint)
def randint(low, high=None, size=None, dtype=int):
  if high is None:
    high = low
    low = 0
  high = jnp.asarray(high)
  low = jnp.asarray(low)
  if size is None:
    size = np.broadcast(low, high).shape

  return JaxArray(jr.randint(DEFAULT.split_key(), shape=_size2shape(size),
                             minval=low, maxval=high, dtype=dtype))


@wraps(np.random.randn)
def randn(*dn):
  return JaxArray(jr.normal(DEFAULT.split_key(), shape=dn))


@wraps(np.random.random)
def random(size=None):
  return JaxArray(jr.uniform(DEFAULT.split_key(), shape=_size2shape(size), minval=0., maxval=1.))


@wraps(np.random.random_sample)
def random_sample(size=None):
  return JaxArray(jr.uniform(DEFAULT.split_key(), shape=_size2shape(size), minval=0., maxval=1.))


ranf = random_sample
sample = random_sample


@wraps(np.random.choice)
def choice(a, size=None, replace=True, p=None):
  a = a.value if isinstance(a, JaxArray) else a
  if p is not None:
    p = jnp.asarray(p)
  return JaxArray(jr.choice(DEFAULT.split_key(), a=a, shape=_size2shape(size), replace=replace, p=p))


@wraps(np.random.permutation)
def permutation(x):
  x = x.value if isinstance(x, JaxArray) else x
  return JaxArray(jr.permutation(DEFAULT.split_key(), x))


@wraps(np.random.shuffle)
def shuffle(x, axis=0):
  assert isinstance(x, JaxArray), f'Must be a JaxArray, but got {type(x)}'
  x.value = jr.permutation(DEFAULT.split_key(), x.value, axis=axis)


@wraps(np.random.beta)
def beta(a, b, size=None):
  a = jnp.asarray(a)
  b = jnp.asarray(b)
  return JaxArray(jr.beta(DEFAULT.split_key(), a=a, b=b, shape=_size2shape(size)))


@wraps(np.random.exponential)
def exponential(scale=1.0, size=None):
  scale = jnp.asarray(scale)
  return JaxArray(jr.exponential(DEFAULT.split_key(), shape=_size2shape(size)) / scale)


@wraps(np.random.gamma)
def gamma(shape, scale=1.0, size=None):
  shape = jnp.asarray(shape)
  scale = jnp.asarray(scale)
  return JaxArray(jr.gamma(DEFAULT.split_key(), a=shape, shape=_size2shape(size)) * scale)


@wraps(np.random.gumbel)
def gumbel(loc=0.0, scale=1.0, size=None):
  loc = jnp.asarray(loc)
  scale = jnp.asarray(scale)
  return JaxArray(jr.gumbel(DEFAULT.split_key(), shape=_size2shape(size)) * scale + loc)


@wraps(np.random.laplace)
def laplace(loc=0.0, scale=1.0, size=None):
  loc = jnp.asarray(loc)
  scale = jnp.asarray(scale)
  return JaxArray(jr.laplace(DEFAULT.split_key(), shape=_size2shape(size)) * scale + loc)


@wraps(np.random.logistic)
def logistic(loc=0.0, scale=1.0, size=None):
  loc = jnp.asarray(loc)
  scale = jnp.asarray(scale)
  return JaxArray(jr.logistic(DEFAULT.split_key(), shape=_size2shape(size)) * scale + loc)


@wraps(np.random.normal)
def normal(loc=0.0, scale=1.0, size=None):
  loc = jnp.asarray(loc)
  scale = jnp.asarray(scale)
  return JaxArray(jr.normal(DEFAULT.split_key(), shape=_size2shape(size)) * scale + loc)


@wraps(np.random.pareto)
def pareto(a, size=None):
  a = jnp.asarray(a)
  return JaxArray(jr.pareto(DEFAULT.split_key(), b=a, shape=_size2shape(size)))


@wraps(np.random.poisson)
def poisson(lam=1.0, size=None):
  lam = jnp.asarray(lam)
  return JaxArray(jr.poisson(DEFAULT.split_key(), lam=lam, shape=_size2shape(size)))


@wraps(np.random.standard_cauchy)
def standard_cauchy(size=None):
  return JaxArray(jr.cauchy(DEFAULT.split_key(), shape=_size2shape(size)))


@wraps(np.random.standard_exponential)
def standard_exponential(size=None):
  return JaxArray(jr.exponential(DEFAULT.split_key(), shape=_size2shape(size)))


@wraps(np.random.standard_gamma)
def standard_gamma(shape, size=None):
  shape = jnp.asarray(shape)
  return JaxArray(jr.gamma(DEFAULT.split_key(), a=shape, shape=_size2shape(size)))


@wraps(np.random.standard_normal)
def standard_normal(size=None):
  return JaxArray(jr.normal(DEFAULT.split_key(), shape=_size2shape(size)))


@wraps(np.random.standard_t)
def standard_t(df, size=None):
  df = jnp.asarray(df)
  return JaxArray(jr.t(DEFAULT.split_key(), df=df, shape=_size2shape(size)))


@wraps(np.random.uniform)
def uniform(low=0.0, high=1.0, size=None):
  low = jnp.asarray(low)
  high = jnp.asarray(high)
  if size is None:
    size = np.broadcast(low, high).shape

  return JaxArray(jr.uniform(DEFAULT.split_key(), shape=_size2shape(size), minval=low, maxval=high))


def truncated_normal(lower, upper, size=None, scale=1.):
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
  lower = jnp.asarray(lower)
  upper = jnp.asarray(upper)
  if size is None:
    size = np.broadcast(lower, upper).shape

  rands = jr.truncated_normal(DEFAULT.split_key(),
                              lower=lower,
                              upper=upper,
                              shape=_size2shape(size))
  return JaxArray(rands * scale)


def bernoulli(p=0.5, size=None):
  """Sample Bernoulli random values with given shape and mean.

  Args:
    p: optional, a float or array of floats for the mean of the random
      variables. Must be broadcast-compatible with ``shape`` and the values
      should be within [0, 1]. Default 0.5.
    size: optional, a tuple of nonnegative integers representing the result
      shape. Must be broadcast-compatible with ``p.shape``. The default (None)
      produces a result shape equal to ``p.shape``.

  Returns:
    A random array with boolean dtype and shape given by ``shape`` if ``shape``
    is not None, or else ``p.shape``.
  """
  p = jnp.asarray(p)
  if jnp.unique(jnp.logical_and(p >= 0, p <= 1)) != jnp.array([True]):
    raise ValueError(r'Bernoulli parameter p should be within [0, 1], but we got {}'.format(p))

  if size is None:
    size = p.shape

  return JaxArray(jr.bernoulli(DEFAULT.split_key(), p=p, shape=_size2shape(size)))


@wraps(np.random.lognormal)
def lognormal(mean=0.0, sigma=1.0, size=None):
  mean = jnp.asarray(mean)
  sigma = jnp.asarray(sigma)
  samples = jr.normal(DEFAULT.split_key(), shape=_size2shape(size))
  samples = samples * sigma + mean
  samples = jnp.exp(samples)
  return JaxArray(samples)


@wraps(np.random.binomial)
def binomial(n, p, size=None):
  if size is None:
    size = np.broadcast(n, p).shape
  size = _size2shape(size)
  d = {'n': n, 'p': p, 'size': size}
  return JaxArray(hcb_call(lambda x: np.random.binomial(n=x['n'], p=x['p'], size=x['size']),
                           d, result_shape=jax.ShapeDtypeStruct(size, int)))


@wraps(np.random.chisquare)
def chisquare(df, size=None):
  if size is None:
    size = np.shape(df)
  size = _size2shape(size)
  d = {'df': df, 'size': size}
  return JaxArray(hcb_call(lambda x: np.random.chisquare(df=x['df'], size=x['size']),
                           d, result_shape=jax.ShapeDtypeStruct(size, float)))


@wraps(np.random.dirichlet)
def dirichlet(alpha, size=None):
  size = _size2shape(size)
  d = {'alpha': alpha, 'size': size}
  output_shape = size + np.shape(alpha)
  return JaxArray(hcb_call(lambda x: np.random.dirichlet(alpha=x['alpha'], size=x['size']),
                           d, result_shape=jax.ShapeDtypeStruct(output_shape, float)))


@wraps(np.random.f)
def f(dfnum, dfden, size=None):
  if size is None:
    size = np.broadcast(dfnum, dfden).shape
  size = _size2shape(size)
  d = {'dfnum': dfnum, 'dfden': dfden, 'size': size}
  return JaxArray(hcb_call(lambda x: np.random.f(dfnum=x['dfnum'], dfden=x['dfden'], size=x['size']),
                           d, result_shape=jax.ShapeDtypeStruct(size, float)))


@wraps(np.random.geometric)
def geometric(p, size=None):
  if size is None:
    size = np.shape(p)
  size = _size2shape(size)
  d = {'p': p, 'size': size}
  return JaxArray(hcb_call(lambda x: np.random.geometric(p=x['p'], size=x['size']),
                           d, result_shape=jax.ShapeDtypeStruct(size, int)))


@wraps(np.random.hypergeometric)
def hypergeometric(ngood, nbad, nsample, size=None):
  if size is None:
    size = np.broadcast(ngood, nbad, nsample).shape
  size = _size2shape(size)
  d = {'ngood': ngood, 'nbad': nbad, 'nsample': nsample, 'size': size}
  return JaxArray(hcb_call(lambda x: np.random.hypergeometric(ngood=x['ngood'], nbad=x['nbad'],
                                                              nsample=x['nsample'], size=x['size']),
                           d, result_shape=jax.ShapeDtypeStruct(size, int)))


@wraps(np.random.logseries)
def logseries(p, size=None):
  if size is None:
    size = np.shape(p)
  size = _size2shape(size)
  d = {'p': p, 'size': size}
  return JaxArray(hcb_call(lambda x: np.random.logseries(p=x['p'], size=x['size']),
                           d, result_shape=jax.ShapeDtypeStruct(size, int)))


@wraps(np.random.multinomial)
def multinomial(n, pvals, size=None):
  size = _size2shape(size)
  d = {'n': n, 'pvals': pvals, 'size': size}
  output_shape = size + np.shape(pvals)
  return JaxArray(hcb_call(lambda x: np.random.multinomial(n=x['n'], pvals=x['pvals'], size=x['size']),
                           d, result_shape=jax.ShapeDtypeStruct(output_shape, int)))


def _packed_multivariate_normal(d):
  candidate_str = ['warn', 'raise', 'ignore']
  selected = np.array([d['warn'], d['raise'], d['ignore']])

  return np.random.multivariate_normal(mean=d['mean'], cov=d['cov'], size=d['size'],
                                       check_valid=candidate_str[np.arange(3)[selected][0]],
                                       tol=d['tol'])

@wraps(np.random.multivariate_normal)
def multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8):
  size = _size2shape(size)

  if not (check_valid == 'warn' or check_valid == 'raise' or check_valid == 'ignore'):
    raise ValueError(r'multivariate_normal argument check_valid should be "warn", "raise", '
                     'or "ignore", but we got {}'.format(check_valid))

  d = {'mean': mean, 'cov': cov, 'size': size,
       'warn': True if check_valid == 'warn' else False,
       'raise': True if check_valid == 'raise' else False,
       'ignore': True if check_valid == 'ignore' else False,
       'tol': tol}
  output_shape = size + np.shape(mean)

  return JaxArray(hcb_call(_packed_multivariate_normal, d,
                           result_shape=jax.ShapeDtypeStruct(output_shape, float)))

