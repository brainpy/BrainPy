# -*- coding: utf-8 -*-
import warnings
from collections import namedtuple
from functools import partial
from operator import index

import jax
import numpy as np
from jax import lax, jit, vmap, numpy as jnp, random as jr, core
from jax._src import dtypes
from jax.experimental.host_callback import call
from jax.tree_util import register_pytree_node

from brainpy.math.jaxarray import JaxArray, Variable
from brainpy.tools.errors import check_error_in_jit
from .utils import wraps

__all__ = [
  'RandomState', 'Generator',

  'seed', 'default_rng',

  'rand', 'randint', 'random_integers', 'randn', 'random',
  'random_sample', 'ranf', 'sample', 'choice', 'permutation', 'shuffle', 'beta',
  'exponential', 'gamma', 'gumbel', 'laplace', 'logistic', 'normal', 'pareto',
  'poisson', 'standard_cauchy', 'standard_exponential', 'standard_gamma',
  'standard_normal', 'standard_t', 'uniform', 'truncated_normal', 'bernoulli',
  'lognormal', 'binomial', 'chisquare', 'dirichlet', 'geometric', 'f',
  'hypergeometric', 'logseries', 'multinomial', 'multivariate_normal',
  'negative_binomial', 'noncentral_chisquare', 'noncentral_f', 'power',
  'rayleigh', 'triangular', 'vonmises', 'wald', 'weibull', 'weibull_min',
  'zipf', 'maxwell', 't', 'orthogonal', 'loggamma', 'categorical',
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
  shape = core.as_named_shape(shape)
  if param_shapes:
    shape_ = lax.broadcast_shapes(shape.positional, *param_shapes)
    if shape.positional != shape_:
      msg = ("{} parameter shapes must be broadcast-compatible with shape "
             "argument, and the result of broadcasting the shapes must equal "
             "the shape argument, but got result {} for shape argument {}.")
      raise ValueError(msg.format(name, shape_, shape))


def _remove_jax_array(a):
  return a.value if isinstance(a, JaxArray) else a


def _const(example, val):
  dtype = dtypes.dtype(example, canonicalize=True)
  if dtypes.is_python_scalar(example):
    val = dtypes.scalar_type_of(example)(val)
    return val if dtype == dtypes.dtype(val, canonicalize=True) else np.array(val, dtype)
  return np.array(val, dtype)


_tr_params = namedtuple(
  "tr_params", ["c", "b", "a", "alpha", "u_r", "v_r", "m", "log_p", "log1_p", "log_h"]
)


def _get_tr_params(n, p):
  # See Table 1. Additionally, we pre-compute log(p), log1(-p) and the
  # constant terms, that depend only on (n, p, m) in log(f(k)) (bottom of page 5).
  mu = n * p
  spq = jnp.sqrt(mu * (1 - p))
  c = mu + 0.5
  b = 1.15 + 2.53 * spq
  a = -0.0873 + 0.0248 * b + 0.01 * p
  alpha = (2.83 + 5.1 / b) * spq
  u_r = 0.43
  v_r = 0.92 - 4.2 / b
  m = jnp.floor((n + 1) * p).astype(n.dtype)
  log_p = jnp.log(p)
  log1_p = jnp.log1p(-p)
  log_h = ((m + 0.5) * (jnp.log((m + 1.0) / (n - m + 1.0)) + log1_p - log_p) +
           _stirling_approx_tail(m) + _stirling_approx_tail(n - m))
  return _tr_params(c, b, a, alpha, u_r, v_r, m, log_p, log1_p, log_h)


def _stirling_approx_tail(k):
  precomputed = jnp.array([0.08106146679532726,
                           0.04134069595540929,
                           0.02767792568499834,
                           0.02079067210376509,
                           0.01664469118982119,
                           0.01387612882307075,
                           0.01189670994589177,
                           0.01041126526197209,
                           0.009255462182712733,
                           0.008330563433362871, ])
  kp1 = k + 1
  kp1sq = (k + 1) ** 2
  return jnp.where(k < 10,
                   precomputed[k],
                   (1.0 / 12 - (1.0 / 360 - (1.0 / 1260) / kp1sq) / kp1sq) / kp1)


def _binomial_btrs(key, p, n):
  """
  Based on the transformed rejection sampling algorithm (BTRS) from the
  following reference:

  Hormann, "The Generation of Binonmial Random Variates"
  (https://core.ac.uk/download/pdf/11007254.pdf)
  """

  def _btrs_body_fn(val):
    _, key, _, _ = val
    key, key_u, key_v = jr.split(key, 3)
    u = jr.uniform(key_u)
    v = jr.uniform(key_v)
    u = u - 0.5
    k = jnp.floor(
      (2 * tr_params.a / (0.5 - jnp.abs(u)) + tr_params.b) * u + tr_params.c
    ).astype(n.dtype)
    return k, key, u, v

  def _btrs_cond_fn(val):
    def accept_fn(k, u, v):
      # See acceptance condition in Step 3. (Page 3) of TRS algorithm
      # v <= f(k) * g_grad(u) / alpha

      m = tr_params.m
      log_p = tr_params.log_p
      log1_p = tr_params.log1_p
      # See: formula for log(f(k)) at bottom of Page 5.
      log_f = (
          (n + 1.0) * jnp.log((n - m + 1.0) / (n - k + 1.0))
          + (k + 0.5) * (jnp.log((n - k + 1.0) / (k + 1.0)) + log_p - log1_p)
          + (_stirling_approx_tail(k) - _stirling_approx_tail(n - k))
          + tr_params.log_h
      )
      g = (tr_params.a / (0.5 - jnp.abs(u)) ** 2) + tr_params.b
      return jnp.log((v * tr_params.alpha) / g) <= log_f

    k, key, u, v = val
    early_accept = (jnp.abs(u) <= tr_params.u_r) & (v <= tr_params.v_r)
    early_reject = (k < 0) | (k > n)
    return lax.cond(
      early_accept | early_reject,
      (),
      lambda _: ~early_accept,
      (k, u, v),
      lambda x: ~accept_fn(*x),
    )

  tr_params = _get_tr_params(n, p)
  ret = lax.while_loop(
    _btrs_cond_fn, _btrs_body_fn, (-1, key, 1.0, 1.0)
  )  # use k=-1 initially so that cond_fn returns True
  return ret[0]


def _binomial_inversion(key, p, n):
  def _binom_inv_body_fn(val):
    i, key, geom_acc = val
    key, key_u = jr.split(key)
    u = jr.uniform(key_u)
    geom = jnp.floor(jnp.log1p(-u) / log1_p) + 1
    geom_acc = geom_acc + geom
    return i + 1, key, geom_acc

  def _binom_inv_cond_fn(val):
    i, _, geom_acc = val
    return geom_acc <= n

  log1_p = jnp.log1p(-p)
  ret = lax.while_loop(_binom_inv_cond_fn, _binom_inv_body_fn, (-1, key, 0.0))
  return ret[0]


def _binomial_dispatch(key, p, n):
  def dispatch(key, p, n):
    is_le_mid = p <= 0.5
    pq = jnp.where(is_le_mid, p, 1 - p)
    mu = n * pq
    k = lax.cond(
      mu < 10,
      (key, pq, n),
      lambda x: _binomial_inversion(*x),
      (key, pq, n),
      lambda x: _binomial_btrs(*x),
    )
    return jnp.where(is_le_mid, k, n - k)

  # Return 0 for nan `p` or negative `n`, since nan values are not allowed for integer types
  cond0 = jnp.isfinite(p) & (n > 0) & (p > 0)
  return lax.cond(
    cond0 & (p < 1),
    (key, p, n),
    lambda x: dispatch(*x),
    (),
    lambda _: jnp.where(cond0, n, 0),
  )


@partial(jit, static_argnums=(3,))
def _binomial(key, p, n, shape):
  shape = shape or lax.broadcast_shapes(jnp.shape(p), jnp.shape(n))
  # reshape to map over axis 0
  p = jnp.reshape(jnp.broadcast_to(p, shape), -1)
  n = jnp.reshape(jnp.broadcast_to(n, shape), -1)
  key = jr.split(key, jnp.size(p))
  if jax.default_backend() == "cpu":
    ret = lax.map(lambda x: _binomial_dispatch(*x), (key, p, n))
  else:
    ret = vmap(lambda *x: _binomial_dispatch(*x))(key, p, n)
  return jnp.reshape(ret, shape)


@partial(jit, static_argnums=(2,))
def _categorical(key, p, shape):
  # this implementation is fast when event shape is small, and slow otherwise
  # Ref: https://stackoverflow.com/a/34190035
  shape = shape or p.shape[:-1]
  s = jnp.cumsum(p, axis=-1)
  r = jr.uniform(key, shape=shape + (1,))
  return jnp.sum(s < r, axis=-1)


def _scatter_add_one(operand, indices, updates):
  return lax.scatter_add(
    operand,
    indices,
    updates,
    lax.ScatterDimensionNumbers(
      update_window_dims=(),
      inserted_window_dims=(0,),
      scatter_dims_to_operand_dims=(0,),
    ),
  )


def _reshape(x, shape):
  if isinstance(x, (int, float, np.ndarray, np.generic)):
    return np.reshape(x, shape)
  else:
    return jnp.reshape(x, shape)


def _promote_shapes(*args, shape=()):
  # adapted from lax.lax_numpy
  if len(args) < 2 and not shape:
    return args
  else:
    shapes = [jnp.shape(arg) for arg in args]
    num_dims = len(lax.broadcast_shapes(shape, *shapes))
    return [
      _reshape(arg, (1,) * (num_dims - len(s)) + s) if len(s) < num_dims else arg
      for arg, s in zip(args, shapes)
    ]


@partial(jit, static_argnums=(3, 4))
def _multinomial(key, p, n, n_max, shape=()):
  if jnp.shape(n) != jnp.shape(p)[:-1]:
    broadcast_shape = lax.broadcast_shapes(jnp.shape(n), jnp.shape(p)[:-1])
    n = jnp.broadcast_to(n, broadcast_shape)
    p = jnp.broadcast_to(p, broadcast_shape + jnp.shape(p)[-1:])
  shape = shape or p.shape[:-1]
  if n_max == 0:
    return jnp.zeros(shape + p.shape[-1:], dtype=jnp.result_type(int))
  # get indices from categorical distribution then gather the result
  indices = _categorical(key, p, (n_max,) + shape)
  # mask out values when counts is heterogeneous
  if jnp.ndim(n) > 0:
    mask = _promote_shapes(jnp.arange(n_max) < jnp.expand_dims(n, -1), shape=shape + (n_max,))[0]
    mask = jnp.moveaxis(mask, -1, 0).astype(indices.dtype)
    excess = jnp.concatenate([jnp.expand_dims(n_max - n, -1),
                              jnp.zeros(jnp.shape(n) + (p.shape[-1] - 1,))],
                             -1)
  else:
    mask = 1
    excess = 0
  # NB: we transpose to move batch shape to the front
  indices_2D = (jnp.reshape(indices * mask, (n_max, -1))).T
  samples_2D = vmap(_scatter_add_one)(jnp.zeros((indices_2D.shape[0], p.shape[-1]), dtype=indices.dtype),
                                      jnp.expand_dims(indices_2D, axis=-1),
                                      jnp.ones(indices_2D.shape, dtype=indices.dtype))
  return jnp.reshape(samples_2D, shape + p.shape[-1:]) - excess


@partial(jit, static_argnums=(2, 3))
def _von_mises_centered(key, concentration, shape, dtype=jnp.float64):
  """Compute centered von Mises samples using rejection sampling from [1]_ with wrapped Cauchy proposal.

  Returns
  -------
  out: array_like
     centered samples from von Mises

  References
  ----------
  .. [1] Luc Devroye "Non-Uniform Random Variate Generation", Springer-Verlag, 1986;
         Chapter 9, p. 473-476. http://www.nrbook.com/devroye/Devroye_files/chapter_nine.pdf

  """
  shape = shape or jnp.shape(concentration)
  dtype = jnp.result_type(dtype)
  concentration = lax.convert_element_type(concentration, dtype)
  concentration = jnp.broadcast_to(concentration, shape)

  s_cutoff_map = {
    jnp.dtype(jnp.float16): 1.8e-1,
    jnp.dtype(jnp.float32): 2e-2,
    jnp.dtype(jnp.float64): 1.2e-4,
  }
  s_cutoff = s_cutoff_map.get(dtype)

  r = 1.0 + jnp.sqrt(1.0 + 4.0 * concentration ** 2)
  rho = (r - jnp.sqrt(2.0 * r)) / (2.0 * concentration)
  s_exact = (1.0 + rho ** 2) / (2.0 * rho)

  s_approximate = 1.0 / concentration

  s = jnp.where(concentration > s_cutoff, s_exact, s_approximate)

  def cond_fn(*args):
    """check if all are done or reached max number of iterations"""
    i, _, done, _, _ = args[0]
    return jnp.bitwise_and(i < 100, jnp.logical_not(jnp.all(done)))

  def body_fn(*args):
    i, key, done, _, w = args[0]
    uni_ukey, uni_vkey, key = jr.split(key, 3)
    u = jr.uniform(
      key=uni_ukey,
      shape=shape,
      dtype=concentration.dtype,
      minval=-1.0,
      maxval=1.0,
    )
    z = jnp.cos(jnp.pi * u)
    w = jnp.where(done, w, (1.0 + s * z) / (s + z))  # Update where not done
    y = concentration * (s - w)
    v = jr.uniform(key=uni_vkey, shape=shape, dtype=concentration.dtype)
    accept = (y * (2.0 - y) >= v) | (jnp.log(y / v) + 1.0 >= y)
    return i + 1, key, accept | done, u, w

  init_done = jnp.zeros(shape, dtype=bool)
  init_u = jnp.zeros(shape)
  init_w = jnp.zeros(shape)

  _, _, done, u, w = lax.while_loop(
    cond_fun=cond_fn,
    body_fun=body_fn,
    init_val=(jnp.array(0), key, init_done, init_u, init_w),
  )

  return jnp.sign(u) * jnp.arccos(w)


def _loc_scale(loc, scale, value):
  if loc is None:
    if scale is None:
      return JaxArray(value)
    else:
      return JaxArray(value * scale)
  else:
    if scale is None:
      return JaxArray(value + loc)
    else:
      return JaxArray(value * scale + loc)


def _check_py_seq(seq):
  return jnp.asarray(seq) if isinstance(seq, (tuple, list)) else seq


class RandomState(Variable):
  """RandomState that track the random generator state. """
  __slots__ = ()

  def __init__(self, seed_or_key=None, seed=None):
    """RandomState constructor.

    Parameters
    ----------
    seed_or_key: int, Array, optional
      It can be an integer for initial seed of the random number generator,
      or it can be a JAX's PRNKey, which is an array with two elements and `uint32` dtype.

      .. versionadded:: 2.2.3.4

    seed : int, Array, optional
      Same as `seed_or_key`.

      .. deprecated:: 2.2.3.4
         Will be removed since version 2.4.
    """
    if seed is not None:
      if seed_or_key is not None:
        raise ValueError('Please set "seed_or_key" or "seed", not both.')
      seed_or_key = seed
      warnings.warn('Please use seed_or_key instead. '
                    'seed will be removed since 2.4.0', UserWarning)

    if seed_or_key is None:
      seed_or_key = np.random.randint(0, 100000, 2, dtype=np.uint32)
    if isinstance(seed_or_key, int):
      key = jr.PRNGKey(seed_or_key)
    else:
      if len(seed_or_key) != 2 and seed_or_key.dtype != np.uint32:
        raise ValueError('key must be an array with dtype uint32. '
                         f'But we got {seed_or_key}')
      key = seed_or_key
    super(RandomState, self).__init__(key)

  # ------------------- #
  # seed and random key #
  # ------------------- #

  def seed(self, seed_or_key=None, seed=None):
    """Sets a new random seed.

    Parameters
    ----------
    seed_or_key: int, Array, optional
      It can be an integer for initial seed of the random number generator,
      or it can be a JAX's PRNKey, which is an array with two elements and `uint32` dtype.

      .. versionadded:: 2.2.3.4

    seed : int, Array, optional
      Same as `seed_or_key`.

      .. deprecated:: 2.2.3.4
         Will be removed since version 2.4.
    """
    if seed is not None:
      if seed_or_key is not None:
        raise ValueError('Please set "seed_or_key" or "seed", not both.')
      seed_or_key = seed
      warnings.warn('Please use seed_or_key instead. '
                    'seed will be removed since 2.4.0', UserWarning)

    if seed_or_key is None:
      seed_or_key = np.random.randint(0, 100000, 2, dtype=np.uint32)
    if isinstance(seed_or_key, int):
      key = jr.PRNGKey(seed_or_key)
    else:
      if len(seed_or_key) != 2 and seed_or_key.dtype != np.uint32:
        raise ValueError('key must be an array with dtype uint32. '
                         f'But we got {seed_or_key}')
      key = seed_or_key
    self.value = key

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

  def rand(self, *dn, key=None):
    key = self.split_key() if key is None else key
    return JaxArray(jr.uniform(key, shape=dn, minval=0., maxval=1.))

  def randint(self, low, high=None, size=None, dtype=jnp.int_, key=None):
    low = _remove_jax_array(low)
    high = _remove_jax_array(high)
    if high is None:
      high = low
      low = 0
    high = _check_py_seq(high)
    low = _check_py_seq(low)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(low),
                                  jnp.shape(high))
    key = self.split_key() if key is None else key
    return JaxArray(jr.randint(key,
                               shape=_size2shape(size),
                               minval=low, maxval=high, dtype=dtype))

  def random_integers(self, low, high=None, size=None, key=None):
    low = _remove_jax_array(low)
    high = _remove_jax_array(high)
    low = _check_py_seq(low)
    high = _check_py_seq(high)
    if high is None:
      high = low
      low = 1
    high += 1
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
    key = self.split_key() if key is None else key
    return JaxArray(jr.randint(key,
                               shape=_size2shape(size),
                               minval=low,
                               maxval=high))

  def randn(self, *dn, key=None):
    key = self.split_key() if key is None else key
    return JaxArray(jr.normal(key, shape=dn))

  def random(self, size=None, key=None):
    key = self.split_key() if key is None else key
    return JaxArray(jr.uniform(key, shape=_size2shape(size), minval=0., maxval=1.))

  def random_sample(self, size=None, key=None):
    return self.random(size=size, key=key)

  def ranf(self, size=None, key=None):
    return self.random(size=size, key=key)

  def sample(self, size=None, key=None):
    return self.random(size=size, key=key)

  def choice(self, a, size=None, replace=True, p=None, key=None):
    a = _remove_jax_array(a)
    p = _remove_jax_array(p)
    a = _check_py_seq(a)
    p = _check_py_seq(p)
    key = self.split_key() if key is None else key
    return JaxArray(jr.choice(key, a=a, shape=_size2shape(size),
                              replace=replace, p=p))

  def permutation(self, x, axis: int = 0, independent: bool = False, key=None):
    x = x.value if isinstance(x, JaxArray) else x
    x = _check_py_seq(x)
    key = self.split_key() if key is None else key
    return JaxArray(jr.permutation(key, x, axis=axis, independent=independent))

  def shuffle(self, x, axis=0, key=None):
    assert isinstance(x, JaxArray), f'Must be a JaxArray, but got {type(x)}'
    key = self.split_key() if key is None else key
    x.value = jr.permutation(key, x.value, axis=axis)

  def beta(self, a, b, size=None, key=None):
    a = a.value if isinstance(a, JaxArray) else a
    b = b.value if isinstance(b, JaxArray) else b
    a = _check_py_seq(a)
    b = _check_py_seq(b)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(a), jnp.shape(b))
    key = self.split_key() if key is None else key
    return JaxArray(jr.beta(key, a=a, b=b, shape=_size2shape(size)))

  def exponential(self, scale=None, size=None, key=None):
    scale = _remove_jax_array(scale)
    scale = _check_py_seq(scale)
    if size is None:
      size = jnp.shape(scale)
    key = self.split_key() if key is None else key
    r = jr.exponential(key, shape=_size2shape(size))
    if scale is None:
      return JaxArray(r)
    else:
      return JaxArray(r / scale)

  def gamma(self, shape, scale=None, size=None, key=None):
    shape = _remove_jax_array(shape)
    scale = _remove_jax_array(scale)
    shape = _check_py_seq(shape)
    scale = _check_py_seq(scale)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(shape), jnp.shape(scale))
    key = self.split_key() if key is None else key
    r = jr.gamma(key, a=shape, shape=_size2shape(size))
    if scale is None:
      return JaxArray(r)
    else:
      return JaxArray(r * scale)

  def gumbel(self, loc=None, scale=None, size=None, key=None):
    loc = _remove_jax_array(loc)
    scale = _remove_jax_array(scale)
    loc = _check_py_seq(loc)
    scale = _check_py_seq(scale)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
    key = self.split_key() if key is None else key
    return _loc_scale(loc, scale, jr.gumbel(key, shape=_size2shape(size)))

  def laplace(self, loc=None, scale=None, size=None, key=None):
    loc = _remove_jax_array(loc)
    scale = _remove_jax_array(scale)
    loc = _check_py_seq(loc)
    scale = _check_py_seq(scale)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
    key = self.split_key() if key is None else key
    return _loc_scale(loc, scale, jr.laplace(key, shape=_size2shape(size)))

  def logistic(self, loc=None, scale=None, size=None, key=None):
    loc = _remove_jax_array(loc)
    scale = _remove_jax_array(scale)
    loc = _check_py_seq(loc)
    scale = _check_py_seq(scale)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
    key = self.split_key() if key is None else key
    return _loc_scale(loc, scale, jr.logistic(key, shape=_size2shape(size)))

  def normal(self, loc=None, scale=None, size=None, key=None):
    loc = _remove_jax_array(loc)
    scale = _remove_jax_array(scale)
    loc = _check_py_seq(loc)
    scale = _check_py_seq(scale)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(scale), jnp.shape(loc))
    key = self.split_key() if key is None else key
    return _loc_scale(loc, scale, jr.normal(key, shape=_size2shape(size)))

  def pareto(self, a, size=None, key=None):
    a = _remove_jax_array(a)
    a = _check_py_seq(a)
    if size is None:
      size = jnp.shape(a)
    key = self.split_key() if key is None else key
    return JaxArray(jr.pareto(key, b=a, shape=_size2shape(size)))

  def poisson(self, lam=1.0, size=None, key=None):
    lam = _check_py_seq(_remove_jax_array(lam))
    if size is None:
      size = jnp.shape(lam)
    key = self.split_key() if key is None else key
    return JaxArray(jr.poisson(key, lam=lam, shape=_size2shape(size)))

  def standard_cauchy(self, size=None, key=None):
    key = self.split_key() if key is None else key
    return JaxArray(jr.cauchy(key, shape=_size2shape(size)))

  def standard_exponential(self, size=None, key=None):
    key = self.split_key() if key is None else key
    return JaxArray(jr.exponential(key, shape=_size2shape(size)))

  def standard_gamma(self, shape, size=None, key=None):
    shape = _remove_jax_array(shape)
    shape = _check_py_seq(shape)
    if size is None:
      size = jnp.shape(shape)
    key = self.split_key() if key is None else key
    return JaxArray(jr.gamma(key, a=shape, shape=_size2shape(size)))

  def standard_normal(self, size=None, key=None):
    key = self.split_key() if key is None else key
    return JaxArray(jr.normal(key, shape=_size2shape(size)))

  def standard_t(self, df, size=None, key=None):
    df = _remove_jax_array(df)
    df = _check_py_seq(df)
    if size is None:
      size = jnp.shape(size)
    key = self.split_key() if key is None else key
    return JaxArray(jr.t(key, df=df, shape=_size2shape(size)))

  def uniform(self, low=0.0, high=1.0, size=None, key=None):
    low = _remove_jax_array(low)
    high = _remove_jax_array(high)
    low = _check_py_seq(low)
    high = _check_py_seq(high)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
    key = self.split_key() if key is None else key
    return JaxArray(jr.uniform(key,
                               shape=_size2shape(size),
                               minval=low,
                               maxval=high))

  def truncated_normal(self, lower, upper, size, scale=None, key=None):
    lower = _remove_jax_array(lower)
    lower = _check_py_seq(lower)
    upper = _remove_jax_array(upper)
    upper = _check_py_seq(upper)
    scale = _remove_jax_array(scale)
    scale = _check_py_seq(scale)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(lower),
                                  jnp.shape(upper),
                                  jnp.shape(scale))
    key = self.split_key() if key is None else key
    rands = jr.truncated_normal(key,
                                lower=lower,
                                upper=upper,
                                shape=_size2shape(size))
    if scale is None:
      return JaxArray(rands)
    else:
      return JaxArray(rands * scale)

  def _check_p(self, p):
    raise ValueError(f'Parameter p should be within [0, 1], but we got {p}')

  def bernoulli(self, p, size=None, key=None):
    p = _check_py_seq(_remove_jax_array(p))
    check_error_in_jit(jnp.any(jnp.logical_and(p < 0, p > 1)), self._check_p, p)
    if size is None:
      size = jnp.shape(p)
    key = self.split_key() if key is None else key
    return JaxArray(jr.bernoulli(key, p=p, shape=_size2shape(size)))

  def lognormal(self, mean=None, sigma=None, size=None, key=None):
    mean = _check_py_seq(_remove_jax_array(mean))
    sigma = _check_py_seq(_remove_jax_array(sigma))
    if size is None:
      size = jnp.broadcast_shapes(jnp.shape(mean),
                                  jnp.shape(sigma))
    key = self.split_key() if key is None else key
    samples = jr.normal(key, shape=_size2shape(size))
    samples = _loc_scale(mean, sigma, samples)
    samples = jnp.exp(samples.value)
    return JaxArray(samples)

  def binomial(self, n, p, size=None, key=None):
    n = _check_py_seq(n.value if isinstance(n, JaxArray) else n)
    p = _check_py_seq(p.value if isinstance(p, JaxArray) else p)
    check_error_in_jit(jnp.any(jnp.logical_and(p < 0, p > 1)), self._check_p, p)
    if size is None:
      size = jnp.broadcast_shapes(jnp.shape(n), jnp.shape(p))
    key = self.split_key() if key is None else key
    return JaxArray(_binomial(key, p, n, shape=_size2shape(size)))

  def chisquare(self, df, size=None, key=None):
    df = _check_py_seq(_remove_jax_array(df))
    key = self.split_key() if key is None else key
    if size is None:
      if jnp.ndim(df) == 0:
        dist = jr.normal(key, (df,)) ** 2
        dist = dist.sum()
      else:
        raise NotImplementedError('Do not support non-scale "df" when "size" is None')
    else:
      dist = jr.normal(key, (df,) + _size2shape(size)) ** 2
      dist = dist.sum(axis=0)
    return JaxArray(dist)

  def dirichlet(self, alpha, size=None, key=None):
    key = self.split_key() if key is None else key
    alpha = _check_py_seq(_remove_jax_array(alpha))
    return JaxArray(jr.dirichlet(key, alpha=alpha, shape=_size2shape(size)))

  def geometric(self, p, size=None, key=None):
    p = _remove_jax_array(p)
    p = _check_py_seq(p)
    if size is None:
      size = jnp.shape(p)
    key = self.split_key() if key is None else key
    u = jr.uniform(key, size)
    r = jnp.floor(jnp.log1p(-u) / jnp.log1p(-p))
    return JaxArray(r)

  def _check_p2(self, p):
    raise ValueError(f'We require `sum(pvals[:-1]) <= 1`. But we got {p}')

  def multinomial(self, n, pvals, size=None, key=None):
    key = self.split_key() if key is None else key
    n = _check_py_seq(_remove_jax_array(n))
    pvals = _check_py_seq(_remove_jax_array(pvals))
    check_error_in_jit(jnp.sum(pvals[:-1]) > 1., self._check_p2, pvals)
    if isinstance(n, jax.core.Tracer):
      raise ValueError("The total count parameter `n` should not be a jax abstract array.")
    size = _size2shape(size)
    n_max = int(np.max(jax.device_get(n)))
    batch_shape = lax.broadcast_shapes(jnp.shape(pvals)[:-1], jnp.shape(n))
    return JaxArray(_multinomial(key, pvals, n, n_max, batch_shape + size))

  def multivariate_normal(self, mean, cov, size=None, method: str = 'cholesky', key=None):
    if method not in {'svd', 'eigh', 'cholesky'}:
      raise ValueError("method must be one of {'svd', 'eigh', 'cholesky'}")
    mean = _check_py_seq(_remove_jax_array(mean))
    cov = _check_py_seq(_remove_jax_array(cov))
    key = self.split_key() if key is None else key

    if not jnp.ndim(mean) >= 1:
      raise ValueError(f"multivariate_normal requires mean.ndim >= 1, got mean.ndim == {jnp.ndim(mean)}")
    if not jnp.ndim(cov) >= 2:
      raise ValueError(f"multivariate_normal requires cov.ndim >= 2, got cov.ndim == {jnp.ndim(cov)}")
    n = mean.shape[-1]
    if jnp.shape(cov)[-2:] != (n, n):
      raise ValueError(f"multivariate_normal requires cov.shape == (..., n, n) for n={n}, "
                       f"but got cov.shape == {jnp.shape(cov)}.")
    if size is None:
      size = lax.broadcast_shapes(mean.shape[:-1], cov.shape[:-2])
    else:
      size = _size2shape(size)
      _check_shape("normal", size, mean.shape[:-1], cov.shape[:-2])

    if method == 'svd':
      (u, s, _) = jnp.linalg.svd(cov)
      factor = u * jnp.sqrt(s[..., None, :])
    elif method == 'eigh':
      (w, v) = jnp.linalg.eigh(cov)
      factor = v * jnp.sqrt(w[..., None, :])
    else:  # 'cholesky'
      factor = jnp.linalg.cholesky(cov)
    normal_samples = jr.normal(key, size + mean.shape[-1:])
    r = mean + jnp.einsum('...ij,...j->...i', factor, normal_samples)
    return JaxArray(r)

  def rayleigh(self, scale=1.0, size=None, key=None):
    scale = _check_py_seq(_remove_jax_array(scale))
    if size is None:
      size = jnp.shape(scale)
    key = self.split_key() if key is None else key
    x = jnp.sqrt(-2. * jnp.log(jr.uniform(key, shape=_size2shape(size), minval=0, maxval=1)))
    return JaxArray(x * scale)

  def triangular(self, size=None, key=None):
    key = self.split_key() if key is None else key
    bernoulli_samples = jr.bernoulli(key, p=0.5, shape=_size2shape(size))
    return JaxArray(2 * bernoulli_samples - 1)

  def vonmises(self, mu, kappa, size=None, key=None):
    key = self.split_key() if key is None else key
    mu = _check_py_seq(_remove_jax_array(mu))
    kappa = _check_py_seq(_remove_jax_array(kappa))
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(mu), jnp.shape(kappa))
    size = _size2shape(size)
    samples = _von_mises_centered(key, kappa, size)
    samples = samples + mu
    samples = (samples + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    return JaxArray(samples)

  def weibull(self, a, size=None, key=None):
    key = self.split_key() if key is None else key
    a = _check_py_seq(_remove_jax_array(a))
    if size is None:
      size = jnp.shape(a)
    else:
      if jnp.size(a) > 1:
        raise ValueError(f'"a" should be a scalar when "size" is provided. But we got {a}')
    size = _size2shape(size)
    random_uniform = jr.uniform(key=key, shape=size, minval=0, maxval=1)
    r = jnp.power(-jnp.log1p(-random_uniform), 1.0 / a)
    return JaxArray(r)

  def weibull_min(self, a, scale=None, size=None, key=None):
    """Sample from a Weibull minimum distribution.

    Parameters
    ----------
    a: float, array_like
      The concentration parameter of the distribution.
    scale: float, array_like
      The scale parameter of the distribution.
    size: optional, int, tuple of int
      The shape added to the parameters loc and scale broadcastable shape.

    Returns
    -------
    out: array_like
      The sampling results.
    """
    key = self.split_key() if key is None else key
    a = _check_py_seq(_remove_jax_array(a))
    scale = _check_py_seq(_remove_jax_array(scale))
    if size is None:
      size = jnp.broadcast_shapes(jnp.shape(a), jnp.shape(scale))
    else:
      if jnp.size(a) > 1:
        raise ValueError(f'"a" should be a scalar when "size" is provided. But we got {a}')
    size = _size2shape(size)
    random_uniform = jr.uniform(key=key, shape=size, minval=0, maxval=1)
    r = jnp.power(-jnp.log1p(-random_uniform), 1.0 / a)
    if scale is not None:
      r /= scale
    return JaxArray(r)

  def maxwell(self, size=None, key=None):
    key = self.split_key() if key is None else key
    shape = core.canonicalize_shape(_size2shape(size)) + (3,)
    norm_rvs = jr.normal(key=key, shape=shape)
    return JaxArray(jnp.linalg.norm(norm_rvs, axis=-1))

  def negative_binomial(self, n, p, size=None, key=None):
    n = _check_py_seq(_remove_jax_array(n))
    p = _check_py_seq(_remove_jax_array(p))
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(n), jnp.shape(p))
    size = _size2shape(size)
    logits = jnp.log(p) - jnp.log1p(-p)
    if key is None:
      keys = self.split_keys(2)
    else:
      keys = jr.split(key, 2)
    rate = self.gamma(shape=n, scale=jnp.exp(-logits), size=size, key=keys[0])
    return JaxArray(self.poisson(lam=rate, key=keys[1]))

  def wald(self, mean, scale, size=None, key=None):
    mean = _check_py_seq(_remove_jax_array(mean))
    scale = _check_py_seq(_remove_jax_array(scale))
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(mean), jnp.shape(scale))
    size = _size2shape(size)
    sampled_chi2 = jnp.square(self.randn(*size).value)
    sampled_uniform = self.uniform(size=size, key=key).value
    # Wikipedia defines an intermediate x with the formula
    #   x = loc + loc ** 2 * y / (2 * conc) - loc / (2 * conc) * sqrt(4 * loc * conc * y + loc ** 2 * y ** 2)
    # where y ~ N(0, 1)**2 (sampled_chi2 above) and conc is the concentration.
    # Let us write
    #   w = loc * y / (2 * conc)
    # Then we can extract the common factor in the last two terms to obtain
    #   x = loc + loc * w * (1 - sqrt(2 / w + 1))
    # Now we see that the Wikipedia formula suffers from catastrphic
    # cancellation for large w (e.g., if conc << loc).
    #
    # Fortunately, we can fix this by multiplying both sides
    # by 1 + sqrt(2 / w + 1).  We get
    #   x * (1 + sqrt(2 / w + 1)) =
    #     = loc * (1 + sqrt(2 / w + 1)) + loc * w * (1 - (2 / w + 1))
    #     = loc * (sqrt(2 / w + 1) - 1)
    # The term sqrt(2 / w + 1) + 1 no longer presents numerical
    # difficulties for large w, and sqrt(2 / w + 1) - 1 is just
    # sqrt1pm1(2 / w), which we know how to compute accurately.
    # This just leaves the matter of small w, where 2 / w may
    # overflow.  In the limit a w -> 0, x -> loc, so we just mask
    # that case.
    sqrt1pm1_arg = 4 * scale / (mean * sampled_chi2)  # 2 / w above
    safe_sqrt1pm1_arg = jnp.where(sqrt1pm1_arg < np.inf, sqrt1pm1_arg, 1.0)
    denominator = 1.0 + jnp.sqrt(safe_sqrt1pm1_arg + 1.0)
    ratio = jnp.expm1(0.5 * jnp.log1p(safe_sqrt1pm1_arg)) / denominator
    sampled = mean * jnp.where(sqrt1pm1_arg < np.inf, ratio, 1.0)  # x above
    res = jnp.where(sampled_uniform <= mean / (mean + sampled),
                    sampled,
                    jnp.square(mean) / sampled)
    return JaxArray(res)

  def t(self, df, size=None, key=None):
    df = _check_py_seq(_remove_jax_array(df))
    if size is None:
      size = np.shape(df)
    else:
      size = _size2shape(size)
      _check_shape("t", size, np.shape(df))
    if key is None:
      keys = self.split_keys(2)
    else:
      keys = jr.split(key, 2)
    n = jr.normal(keys[0], size)
    two = _const(n, 2)
    half_df = lax.div(df, two)
    g = jr.gamma(keys[1], half_df, size)
    return JaxArray(n * jnp.sqrt(half_df / g))

  def orthogonal(self, n: int, size=None, key=None):
    key = self.split_key() if key is None else key
    size = _size2shape(size)
    _check_shape("orthogonal", size)
    n = core.concrete_or_error(index, n, "The error occurred in jax.random.orthogonal()")
    z = jr.normal(key, size + (n, n))
    q, r = jnp.linalg.qr(z)
    d = jnp.diagonal(r, 0, -2, -1)
    return JaxArray(q * jnp.expand_dims(d / abs(d), -2))

  def noncentral_chisquare(self, df, nonc, size=None, key=None):
    df = _check_py_seq(_remove_jax_array(df))
    nonc = _check_py_seq(_remove_jax_array(nonc))
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(df), jnp.shape(nonc))
    size = _size2shape(size)
    if key is None:
      keys = self.split_keys(3)
    else:
      keys = jr.split(key, 3)
    i = jr.poisson(keys[0], 0.5 * nonc, shape=size)
    n = jr.normal(keys[1], shape=size) + jnp.sqrt(nonc)
    cond = jnp.greater(df, 1.0)
    df2 = jnp.where(cond, df - 1.0, df + 2.0 * i)
    chi2 = 2.0 * jr.gamma(keys[2], 0.5 * df2, shape=size)
    return JaxArray(jnp.where(cond, chi2 + n * n, chi2))

  def loggamma(self, a, size=None, key=None):
    key = self.split_key() if key is None else key
    a = _check_py_seq(_remove_jax_array(a))
    if size is None:
      size = jnp.shape(a)
    return JaxArray(jr.loggamma(key, a, shape=_size2shape(size)))

  def categorical(self, logits, axis: int = -1, size=None, key=None):
    key = self.split_key() if key is None else key
    logits = _check_py_seq(_remove_jax_array(logits))
    if size is None:
      size = list(jnp.shape(logits))
      size.pop(axis)
    return JaxArray(jr.categorical(key, logits, axis=axis, shape=_size2shape(size)))

  def zipf(self, a, size=None, key=None):
    a = _check_py_seq(_remove_jax_array(a))
    if size is None:
      size = jnp.shape(a)
    return JaxArray(call(lambda x: np.random.zipf(x, size),
                         a,
                         result_shape=jax.ShapeDtypeStruct(size, jnp.int_)))

  def power(self, a, size=None, key=None):
    a = _check_py_seq(_remove_jax_array(a))
    if size is None:
      size = jnp.shape(a)
    size = _size2shape(size)
    return JaxArray(call(lambda a: np.random.power(a=a, size=size),
                         a, result_shape=jax.ShapeDtypeStruct(size, jnp.float_)))

  def f(self, dfnum, dfden, size=None, key=None):
    dfnum = _remove_jax_array(dfnum)
    dfden = _remove_jax_array(dfden)
    dfnum = _check_py_seq(dfnum)
    dfden = _check_py_seq(dfden)
    if size is None:
      size = jnp.broadcast_shapes(jnp.shape(dfnum), jnp.shape(dfden))
    size = _size2shape(size)
    d = {'dfnum': dfnum, 'dfden': dfden}
    return JaxArray(call(lambda x: np.random.f(dfnum=x['dfnum'],
                                               dfden=x['dfden'],
                                               size=size),
                         d,
                         result_shape=jax.ShapeDtypeStruct(size, jnp.float_)))

  def hypergeometric(self, ngood, nbad, nsample, size=None, key=None):
    ngood = _check_py_seq(_remove_jax_array(ngood))
    nbad = _check_py_seq(_remove_jax_array(nbad))
    nsample = _check_py_seq(_remove_jax_array(nsample))

    if size is None:
      size = lax.broadcast_shapes(jnp.shape(ngood),
                                  jnp.shape(nbad),
                                  jnp.shape(nsample))
    size = _size2shape(size)
    d = {'ngood': ngood, 'nbad': nbad, 'nsample': nsample}
    return JaxArray(call(lambda x: np.random.hypergeometric(ngood=x['ngood'],
                                                            nbad=x['nbad'],
                                                            nsample=x['nsample'],
                                                            size=size),
                         d, result_shape=jax.ShapeDtypeStruct(size, jnp.int_)))

  def logseries(self, p, size=None, key=None):
    p = _check_py_seq(_remove_jax_array(p))
    if size is None:
      size = jnp.shape(p)
    size = _size2shape(size)
    return JaxArray(call(lambda p: np.random.logseries(p=p, size=size),
                         p, result_shape=jax.ShapeDtypeStruct(size, jnp.int_)))

  def noncentral_f(self, dfnum, dfden, nonc, size=None, key=None):
    dfnum = _check_py_seq(_remove_jax_array(dfnum))
    dfden = _check_py_seq(_remove_jax_array(dfden))
    nonc = _check_py_seq(_remove_jax_array(nonc))
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(dfnum),
                                  jnp.shape(dfden),
                                  jnp.shape(nonc))
    size = _size2shape(size)
    d = {'dfnum': dfnum, 'dfden': dfden, 'nonc': nonc}
    return JaxArray(call(lambda x: np.random.noncentral_f(dfnum=x['dfnum'],
                                                          dfden=x['dfden'],
                                                          nonc=x['nonc'],
                                                          size=size),
                         d, result_shape=jax.ShapeDtypeStruct(size, jnp.float_)))


# alias
Generator = RandomState

# register pytree
register_pytree_node(RandomState,
                     lambda t: ((t.value,), None),
                     lambda aux_data, flat_contents: RandomState(*flat_contents))

# default random generator
__a = JaxArray(None)
__a._value = np.random.randint(0, 10000, size=2, dtype=np.uint32)
DEFAULT = RandomState(__a)
del __a


@wraps(np.random.default_rng)
def default_rng(seed=None):
  return RandomState(seed)


@wraps(np.random.seed)
def seed(seed=None):
  if seed is None: seed = np.random.randint(0, 100000)
  DEFAULT.seed(seed)
  np.random.seed(seed)


@wraps(np.random.rand)
def rand(*dn, key=None):
  return DEFAULT.rand(*dn, key=key)


@wraps(np.random.randint)
def randint(low, high=None, size=None, dtype=jnp.int_, key=None):
  return DEFAULT.randint(low, high=high, size=size, dtype=dtype, key=key)


@wraps(np.random.random_integers)
def random_integers(low, high=None, size=None, key=None):
  return DEFAULT.random_integers(low, high=high, size=size, key=key)


@wraps(np.random.randn)
def randn(*dn, key=None):
  return DEFAULT.randn(*dn, key=key)


@wraps(np.random.random)
def random(size=None, key=None):
  return DEFAULT.random(size, key=key)


@wraps(np.random.random_sample)
def random_sample(size=None, key=None):
  return DEFAULT.random_sample(size, key=key)


@wraps(np.random.ranf)
def ranf(size=None, key=None):
  return DEFAULT.ranf(size, key=key)


@wraps(np.random.sample)
def sample(size=None, key=None):
  return DEFAULT.sample(size, key=key)


@wraps(np.random.choice)
def choice(a, size=None, replace=True, p=None, key=None):
  a = _remove_jax_array(a)
  return DEFAULT.choice(a=a, size=size, replace=replace, p=p, key=key)


@wraps(np.random.permutation)
def permutation(x, axis: int = 0, independent: bool = False, key=None):
  return DEFAULT.permutation(x, axis=axis, independent=independent, key=key)


@wraps(np.random.shuffle)
def shuffle(x, axis=0, key=None):
  DEFAULT.shuffle(x, axis, key=key)


@wraps(np.random.beta)
def beta(a, b, size=None, key=None):
  return DEFAULT.beta(a, b, size=size, key=key)


@wraps(np.random.exponential)
def exponential(scale=None, size=None, key=None):
  return DEFAULT.exponential(scale, size, key=key)


@wraps(np.random.gamma)
def gamma(shape, scale=None, size=None, key=None):
  return DEFAULT.gamma(shape, scale, size=size, key=key)


@wraps(np.random.gumbel)
def gumbel(loc=None, scale=None, size=None, key=None):
  return DEFAULT.gumbel(loc, scale, size=size, key=key)


@wraps(np.random.laplace)
def laplace(loc=None, scale=None, size=None, key=None):
  return DEFAULT.laplace(loc, scale, size, key=key)


@wraps(np.random.logistic)
def logistic(loc=None, scale=None, size=None, key=None):
  return DEFAULT.logistic(loc, scale, size, key=key)


@wraps(np.random.normal)
def normal(loc=None, scale=None, size=None, key=None):
  return DEFAULT.normal(loc, scale, size, key=key)


@wraps(np.random.pareto)
def pareto(a, size=None, key=None):
  return DEFAULT.pareto(a, size, key=key)


@wraps(np.random.poisson)
def poisson(lam=1.0, size=None, key=None):
  return DEFAULT.poisson(lam, size, key=key)


@wraps(np.random.standard_cauchy)
def standard_cauchy(size=None, key=None):
  return DEFAULT.standard_cauchy(size, key=key)


@wraps(np.random.standard_exponential)
def standard_exponential(size=None, key=None):
  return DEFAULT.standard_exponential(size, key=key)


@wraps(np.random.standard_gamma)
def standard_gamma(shape, size=None, key=None):
  return DEFAULT.standard_gamma(shape, size, key=key)


@wraps(np.random.standard_normal)
def standard_normal(size=None, key=None):
  return DEFAULT.standard_normal(size, key=key)


@wraps(np.random.standard_t)
def standard_t(df, size=None, key=None):
  return DEFAULT.standard_t(df, size, key=key)


@wraps(np.random.uniform)
def uniform(low=0.0, high=1.0, size=None, key=None):
  return DEFAULT.uniform(low, high, size, key=key)


@wraps(jr.truncated_normal)
def truncated_normal(lower, upper, size=None, scale=None, key=None):
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
  return DEFAULT.truncated_normal(lower, upper, size, scale, key=key)


@wraps(jr.bernoulli)
def bernoulli(p=0.5, size=None, key=None):
  """Sample Bernoulli random values with given shape and mean.

  Parameters
  ----------
  p: float, array_like, optional
    A float or array of floats for the mean of the random
    variables. Must be broadcast-compatible with ``shape`` and the values
    should be within [0, 1]. Default 0.5.
  size: optional, tuple of int, int
    A tuple of nonnegative integers representing the result
    shape. Must be broadcast-compatible with ``p.shape``. The default (None)
    produces a result shape equal to ``p.shape``.

  Returns
  -------
  out: array_like
    A random array with boolean dtype and shape given by ``shape`` if ``shape``
    is not None, or else ``p.shape``.
  """
  return DEFAULT.bernoulli(p, size, key=key)


@wraps(np.random.lognormal)
def lognormal(mean=None, sigma=None, size=None, key=None):
  return DEFAULT.lognormal(mean, sigma, size, key=key)


@wraps(np.random.binomial)
def binomial(n, p, size=None, key=None):
  return DEFAULT.binomial(n, p, size, key=key)


@wraps(np.random.chisquare)
def chisquare(df, size=None, key=None):
  return DEFAULT.chisquare(df, size, key=key)


@wraps(np.random.dirichlet)
def dirichlet(alpha, size=None, key=None):
  return DEFAULT.dirichlet(alpha, size, key=key)


@wraps(np.random.geometric)
def geometric(p, size=None, key=None):
  return DEFAULT.geometric(p, size, key=key)


@wraps(np.random.f)
def f(dfnum, dfden, size=None, key=None):
  return DEFAULT.f(dfnum, dfden, size, key=key)


@wraps(np.random.hypergeometric)
def hypergeometric(ngood, nbad, nsample, size=None, key=None):
  return DEFAULT.hypergeometric(ngood, nbad, nsample, size, key=key)


@wraps(np.random.logseries)
def logseries(p, size=None, key=None):
  return DEFAULT.logseries(p, size, key=key)


@wraps(np.random.multinomial)
def multinomial(n, pvals, size=None, key=None):
  return DEFAULT.multinomial(n, pvals, size, key=key)


@wraps(np.random.multivariate_normal)
def multivariate_normal(mean, cov, size=None, method: str = 'cholesky', key=None):
  return DEFAULT.multivariate_normal(mean, cov, size, method, key=key)


@wraps(np.random.negative_binomial)
def negative_binomial(n, p, size=None, key=None):
  return DEFAULT.negative_binomial(n, p, size, key=key)


@wraps(np.random.noncentral_chisquare)
def noncentral_chisquare(df, nonc, size=None, key=None):
  return DEFAULT.noncentral_chisquare(df, nonc, size, key=key)


@wraps(np.random.noncentral_f)
def noncentral_f(dfnum, dfden, nonc, size=None, key=None):
  return DEFAULT.noncentral_f(dfnum, dfden, nonc, size, key=key)


@wraps(np.random.power)
def power(a, size=None, key=None):
  return DEFAULT.power(a, size, key=key)


@wraps(np.random.rayleigh)
def rayleigh(scale=1.0, size=None, key=None):
  return DEFAULT.rayleigh(scale, size, key=key)


@wraps(np.random.triangular)
def triangular(size=None, key=None):
  return DEFAULT.triangular(size, key=key)


@wraps(np.random.vonmises)
def vonmises(mu, kappa, size=None, key=None):
  return DEFAULT.vonmises(mu, kappa, size, key=key)


@wraps(np.random.wald)
def wald(mean, scale, size=None, key=None):
  return DEFAULT.wald(mean, scale, size, key=key)


@wraps(np.random.weibull)
def weibull(a, size=None, key=None):
  return DEFAULT.weibull(a, size, key=key)


@wraps(jr.weibull_min)
def weibull_min(a, scale=None, size=None, key=None):
  return DEFAULT.weibull_min(a, scale, size, key=key)


@wraps(np.random.zipf)
def zipf(a, size=None, key=None):
  return DEFAULT.zipf(a, size, key=key)


@wraps(jr.maxwell)
def maxwell(size=None, key=None):
  return DEFAULT.maxwell(size, key=key)


def t(df, size=None, key=None):
  """Sample Student’s t random values.

  Parameters
  ----------
  df: float, array_like
    A float or array of floats broadcast-compatible with shape representing the parameter of the distribution.
  size: optional, int, tuple of int
    A tuple of non-negative integers specifying the result shape.
    Must be broadcast-compatible with `df`. The default (None) produces a result shape equal to `df.shape`.

  Returns
  -------
  out: array_like
    The sampled value.
  """
  return DEFAULT.t(df, size, key=key)


def orthogonal(n: int, size=None, key=None):
  """Sample uniformly from the orthogonal group `O(n)`.

  Parameters
  ----------
  n: int
     An integer indicating the resulting dimension.
  size: optional, int, tuple of int
    The batch dimensions of the result.

  Returns
  -------
  out: JaxArray
    The sampled results.
  """
  return DEFAULT.orthogonal(n, size, key=key)


def loggamma(a, size=None, key=None):
  """Sample log-gamma random values.

  Parameters
  ----------
  a: float, array_like
    A float or array of floats broadcast-compatible with shape representing the parameter of the distribution.
  size: optional, int, tuple of int
    A tuple of nonnegative integers specifying the result shape.
    Must be broadcast-compatible with `a`. The default (None) produces a result shape equal to `a.shape`.

  Returns
  -------
  out: array_like
    The sampled results.
  """
  return DEFAULT.loggamma(a, size)


@wraps(jr.categorical)
def categorical(logits, axis: int = -1, size=None, key=None):
  return DEFAULT.categorical(logits, axis, size, key=key)
