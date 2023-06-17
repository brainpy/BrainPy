# -*- coding: utf-8 -*-

import warnings
from collections import namedtuple
from functools import partial
from operator import index
from typing import Optional, Union

import jax
import numpy as np
from jax import lax, jit, vmap, numpy as jnp, random as jr, core, dtypes
from jax.experimental.host_callback import call
from jax.tree_util import register_pytree_node_class

from brainpy.check import jit_error
from .compat_numpy import shape
from .environment import get_int
from .ndarray import Array, _return
from .object_transform.variables import Variable

__all__ = [
  'RandomState', 'Generator', 'DEFAULT',

  'seed', 'default_rng', 'split_key',

  # numpy compatibility
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

  # pytorch compatibility
  'rand_like', 'randint_like', 'randn_like',
]


def _formalize_key(key):
  if isinstance(key, int):
    return jr.PRNGKey(key)
  elif isinstance(key, (Array, jnp.ndarray, np.ndarray)):
    if key.dtype != jnp.uint32:
      raise TypeError('key must be a int or an array with two uint32.')
    if key.size != 2:
      raise TypeError('key must be a int or an array with two uint32.')
    return jnp.asarray(key)
  else:
    raise TypeError('key must be a int or an array with two uint32.')


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


def _as_jax_array(a):
  return a.value if isinstance(a, Array) else a


def _is_python_scalar(x):
  if hasattr(x, 'aval'):
    return x.aval.weak_type
  elif np.ndim(x) == 0:
    return True
  elif isinstance(x, (bool, int, float, complex)):
    return True
  else:
    return False


python_scalar_dtypes = {
  bool: np.dtype('bool'),
  int: np.dtype('int64'),
  float: np.dtype('float64'),
  complex: np.dtype('complex128'),
}


def _dtype(x, *, canonicalize: bool = False):
  """Return the dtype object for a value or type, optionally canonicalized based on X64 mode."""
  if x is None:
    raise ValueError(f"Invalid argument to dtype: {x}.")
  elif isinstance(x, type) and x in python_scalar_dtypes:
    dt = python_scalar_dtypes[x]
  elif type(x) in python_scalar_dtypes:
    dt = python_scalar_dtypes[type(x)]
  elif jax.core.is_opaque_dtype(getattr(x, 'dtype', None)):
    dt = x.dtype
  else:
    dt = np.result_type(x)
  return dtypes.canonicalize_dtype(dt) if canonicalize else dt


def _const(example, val):
  if _is_python_scalar(example):
    dtype = dtypes.canonicalize_dtype(type(example))
    val = dtypes.scalar_type_of(example)(val)
    return val if dtype == _dtype(val, canonicalize=True) else np.array(val, dtype)
  else:
    dtype = dtypes.canonicalize_dtype(example.dtype)
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
      return value
    else:
      return value * scale
  else:
    if scale is None:
      return value + loc
    else:
      return value * scale + loc


def _check_py_seq(seq):
  return jnp.asarray(seq) if isinstance(seq, (tuple, list)) else seq


@register_pytree_node_class
class RandomState(Variable):
  """RandomState that track the random generator state. """
  __slots__ = ()

  def __init__(
      self,
      seed_or_key: Optional[Union[int, Array, jax.Array, np.ndarray]] = None,
      seed: Optional[int] = None,
      _ready_to_trace: bool = True,
  ):
    """RandomState constructor.

    Parameters
    ----------
    seed_or_key: int, Array, optional
      It can be an integer for initial seed of the random number generator,
      or it can be a JAX's PRNKey, which is an array with two elements and `uint32` dtype.

      .. versionadded:: 2.2.3.4

    seed : int, ArrayType, optional
      Same as `seed_or_key`.

      .. deprecated:: 2.2.3.4
         Will be removed since version 2.4.
    """
    if seed is not None:
      if seed_or_key is not None:
        raise ValueError('Please set "seed_or_key" or "seed", not both.')
      seed_or_key = seed
      warnings.warn('Please use `seed_or_key` instead. '
                    'seed will be removed since 2.4.0', UserWarning)

    with jax.ensure_compile_time_eval():
      if seed_or_key is None:
        seed_or_key = np.random.randint(0, 100000, 2, dtype=np.uint32)
    if isinstance(seed_or_key, int):
      key = jr.PRNGKey(seed_or_key)
    else:
      if len(seed_or_key) != 2 and seed_or_key.dtype != np.uint32:
        raise ValueError('key must be an array with dtype uint32. '
                         f'But we got {seed_or_key}')
      key = seed_or_key
    super(RandomState, self).__init__(key, _ready_to_trace=_ready_to_trace)

  def __repr__(self) -> str:
    print_code = repr(self.value)
    i = print_code.index('(')
    name = self.__class__.__name__
    return f'{name}(key={print_code[i:]})'

  # ------------------- #
  # seed and random key #
  # ------------------- #

  def clone(self):
    return type(self)(self.split_key())

  def seed(self, seed_or_key=None, seed=None):
    """Sets a new random seed.

    Parameters
    ----------
    seed_or_key: int, ArrayType, optional
      It can be an integer for initial seed of the random number generator,
      or it can be a JAX's PRNKey, which is an array with two elements and `uint32` dtype.

      .. versionadded:: 2.2.3.4

    seed : int, ArrayType, optional
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
    self._value = key

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
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.uniform(key, shape=dn, minval=0., maxval=1.)
    return _return(r)

  def randint(self, low, high=None, size=None, dtype=None, key=None):
    dtype = get_int() if dtype is None else dtype
    low = _as_jax_array(low)
    high = _as_jax_array(high)
    if high is None:
      high = low
      low = 0
    high = _check_py_seq(high)
    low = _check_py_seq(low)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(low),
                                  jnp.shape(high))
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.randint(key,
                   shape=_size2shape(size),
                   minval=low, maxval=high, dtype=dtype)
    return _return(r)

  def random_integers(self, low, high=None, size=None, key=None):
    low = _as_jax_array(low)
    high = _as_jax_array(high)
    low = _check_py_seq(low)
    high = _check_py_seq(high)
    if high is None:
      high = low
      low = 1
    high += 1
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.randint(key,
                   shape=_size2shape(size),
                   minval=low,
                   maxval=high)
    return _return(r)

  def randn(self, *dn, key=None):
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.normal(key, shape=dn)
    return _return(r)

  def random(self, size=None, key=None):
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.uniform(key, shape=_size2shape(size), minval=0., maxval=1.)
    return _return(r)

  def random_sample(self, size=None, key=None):
    r = self.random(size=size, key=key)
    return _return(r)

  def ranf(self, size=None, key=None):
    r = self.random(size=size, key=key)
    return _return(r)

  def sample(self, size=None, key=None):
    r = self.random(size=size, key=key)
    return _return(r)

  def choice(self, a, size=None, replace=True, p=None, key=None):
    a = _as_jax_array(a)
    p = _as_jax_array(p)
    a = _check_py_seq(a)
    p = _check_py_seq(p)
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.choice(key, a=a, shape=_size2shape(size), replace=replace, p=p)
    return _return(r)

  def permutation(self, x, axis: int = 0, independent: bool = False, key=None):
    x = x.value if isinstance(x, Array) else x
    x = _check_py_seq(x)
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.permutation(key, x, axis=axis, independent=independent)
    return _return(r)

  def shuffle(self, x, axis=0, key=None):
    if not isinstance(x, Array):
      raise TypeError('This numpy operator needs in-place updating, therefore '
                      'inputs should be brainpy Array.')
    key = self.split_key() if key is None else _formalize_key(key)
    x.value = jr.permutation(key, x.value, axis=axis)

  def beta(self, a, b, size=None, key=None):
    a = a.value if isinstance(a, Array) else a
    b = b.value if isinstance(b, Array) else b
    a = _check_py_seq(a)
    b = _check_py_seq(b)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(a), jnp.shape(b))
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.beta(key, a=a, b=b, shape=_size2shape(size))
    return _return(r)

  def exponential(self, scale=None, size=None, key=None):
    scale = _as_jax_array(scale)
    scale = _check_py_seq(scale)
    if size is None:
      size = jnp.shape(scale)
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.exponential(key, shape=_size2shape(size))
    if scale is not None:
      r = r / scale
    return _return(r)

  def gamma(self, shape, scale=None, size=None, key=None):
    shape = _as_jax_array(shape)
    scale = _as_jax_array(scale)
    shape = _check_py_seq(shape)
    scale = _check_py_seq(scale)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(shape), jnp.shape(scale))
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.gamma(key, a=shape, shape=_size2shape(size))
    if scale is not None:
      r = r * scale
    return _return(r)

  def gumbel(self, loc=None, scale=None, size=None, key=None):
    loc = _as_jax_array(loc)
    scale = _as_jax_array(scale)
    loc = _check_py_seq(loc)
    scale = _check_py_seq(scale)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
    key = self.split_key() if key is None else _formalize_key(key)
    r = _loc_scale(loc, scale, jr.gumbel(key, shape=_size2shape(size)))
    return _return(r)

  def laplace(self, loc=None, scale=None, size=None, key=None):
    loc = _as_jax_array(loc)
    scale = _as_jax_array(scale)
    loc = _check_py_seq(loc)
    scale = _check_py_seq(scale)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
    key = self.split_key() if key is None else _formalize_key(key)
    r = _loc_scale(loc, scale, jr.laplace(key, shape=_size2shape(size)))
    return _return(r)

  def logistic(self, loc=None, scale=None, size=None, key=None):
    loc = _as_jax_array(loc)
    scale = _as_jax_array(scale)
    loc = _check_py_seq(loc)
    scale = _check_py_seq(scale)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
    key = self.split_key() if key is None else _formalize_key(key)
    r = _loc_scale(loc, scale, jr.logistic(key, shape=_size2shape(size)))
    return _return(r)

  def normal(self, loc=None, scale=None, size=None, key=None):
    loc = _as_jax_array(loc)
    scale = _as_jax_array(scale)
    loc = _check_py_seq(loc)
    scale = _check_py_seq(scale)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(scale), jnp.shape(loc))
    key = self.split_key() if key is None else _formalize_key(key)
    r = _loc_scale(loc, scale, jr.normal(key, shape=_size2shape(size)))
    return _return(r)

  def pareto(self, a, size=None, key=None):
    a = _as_jax_array(a)
    a = _check_py_seq(a)
    if size is None:
      size = jnp.shape(a)
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.pareto(key, b=a, shape=_size2shape(size))
    return _return(r)

  def poisson(self, lam=1.0, size=None, key=None):
    lam = _check_py_seq(_as_jax_array(lam))
    if size is None:
      size = jnp.shape(lam)
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.poisson(key, lam=lam, shape=_size2shape(size))
    return _return(r)

  def standard_cauchy(self, size=None, key=None):
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.cauchy(key, shape=_size2shape(size))
    return _return(r)

  def standard_exponential(self, size=None, key=None):
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.exponential(key, shape=_size2shape(size))
    return _return(r)

  def standard_gamma(self, shape, size=None, key=None):
    shape = _as_jax_array(shape)
    shape = _check_py_seq(shape)
    if size is None:
      size = jnp.shape(shape)
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.gamma(key, a=shape, shape=_size2shape(size))
    return _return(r)

  def standard_normal(self, size=None, key=None):
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.normal(key, shape=_size2shape(size))
    return _return(r)

  def standard_t(self, df, size=None, key=None):
    df = _as_jax_array(df)
    df = _check_py_seq(df)
    if size is None:
      size = jnp.shape(size)
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.t(key, df=df, shape=_size2shape(size))
    return _return(r)

  def uniform(self, low=0.0, high=1.0, size=None, key=None):
    low = _as_jax_array(low)
    high = _as_jax_array(high)
    low = _check_py_seq(low)
    high = _check_py_seq(high)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.uniform(key, shape=_size2shape(size), minval=low, maxval=high)
    return _return(r)

  def truncated_normal(self, lower, upper, size=None, scale=None, key=None):
    lower = _as_jax_array(lower)
    lower = _check_py_seq(lower)
    upper = _as_jax_array(upper)
    upper = _check_py_seq(upper)
    scale = _as_jax_array(scale)
    scale = _check_py_seq(scale)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(lower),
                                  jnp.shape(upper),
                                  jnp.shape(scale))
    key = self.split_key() if key is None else _formalize_key(key)
    rands = jr.truncated_normal(key,
                                lower=lower,
                                upper=upper,
                                shape=_size2shape(size))
    if scale is not None:
      rands = rands * scale
    return _return(rands)

  def _check_p(self, p):
    raise ValueError(f'Parameter p should be within [0, 1], but we got {p}')

  def bernoulli(self, p, size=None, key=None):
    p = _check_py_seq(_as_jax_array(p))
    jit_error(jnp.any(jnp.logical_and(p < 0, p > 1)), self._check_p, p)
    if size is None:
      size = jnp.shape(p)
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.bernoulli(key, p=p, shape=_size2shape(size))
    return _return(r)

  def lognormal(self, mean=None, sigma=None, size=None, key=None):
    mean = _check_py_seq(_as_jax_array(mean))
    sigma = _check_py_seq(_as_jax_array(sigma))
    if size is None:
      size = jnp.broadcast_shapes(jnp.shape(mean),
                                  jnp.shape(sigma))
    key = self.split_key() if key is None else _formalize_key(key)
    samples = jr.normal(key, shape=_size2shape(size))
    samples = _loc_scale(mean, sigma, samples)
    samples = jnp.exp(samples)
    return _return(samples)

  def binomial(self, n, p, size=None, key=None):
    n = _check_py_seq(n.value if isinstance(n, Array) else n)
    p = _check_py_seq(p.value if isinstance(p, Array) else p)
    jit_error(jnp.any(jnp.logical_and(p < 0, p > 1)), self._check_p, p)
    if size is None:
      size = jnp.broadcast_shapes(jnp.shape(n), jnp.shape(p))
    key = self.split_key() if key is None else _formalize_key(key)
    r = _binomial(key, p, n, shape=_size2shape(size))
    return _return(r)

  def chisquare(self, df, size=None, key=None):
    df = _check_py_seq(_as_jax_array(df))
    key = self.split_key() if key is None else _formalize_key(key)
    if size is None:
      if jnp.ndim(df) == 0:
        dist = jr.normal(key, (df,)) ** 2
        dist = dist.sum()
      else:
        raise NotImplementedError('Do not support non-scale "df" when "size" is None')
    else:
      dist = jr.normal(key, (df,) + _size2shape(size)) ** 2
      dist = dist.sum(axis=0)
    return _return(dist)

  def dirichlet(self, alpha, size=None, key=None):
    key = self.split_key() if key is None else _formalize_key(key)
    alpha = _check_py_seq(_as_jax_array(alpha))
    r = jr.dirichlet(key, alpha=alpha, shape=_size2shape(size))
    return _return(r)

  def geometric(self, p, size=None, key=None):
    p = _as_jax_array(p)
    p = _check_py_seq(p)
    if size is None:
      size = jnp.shape(p)
    key = self.split_key() if key is None else _formalize_key(key)
    u = jr.uniform(key, size)
    r = jnp.floor(jnp.log1p(-u) / jnp.log1p(-p))
    return _return(r)

  def _check_p2(self, p):
    raise ValueError(f'We require `sum(pvals[:-1]) <= 1`. But we got {p}')

  def multinomial(self, n, pvals, size=None, key=None):
    key = self.split_key() if key is None else _formalize_key(key)
    n = _check_py_seq(_as_jax_array(n))
    pvals = _check_py_seq(_as_jax_array(pvals))
    jit_error(jnp.sum(pvals[:-1]) > 1., self._check_p2, pvals)
    if isinstance(n, jax.core.Tracer):
      raise ValueError("The total count parameter `n` should not be a jax abstract array.")
    size = _size2shape(size)
    n_max = int(np.max(jax.device_get(n)))
    batch_shape = lax.broadcast_shapes(jnp.shape(pvals)[:-1], jnp.shape(n))
    r = _multinomial(key, pvals, n, n_max, batch_shape + size)
    return _return(r)

  def multivariate_normal(self, mean, cov, size=None, method: str = 'cholesky', key=None):
    if method not in {'svd', 'eigh', 'cholesky'}:
      raise ValueError("method must be one of {'svd', 'eigh', 'cholesky'}")
    mean = _check_py_seq(_as_jax_array(mean))
    cov = _check_py_seq(_as_jax_array(cov))
    key = self.split_key() if key is None else _formalize_key(key)

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
    return _return(r)

  def rayleigh(self, scale=1.0, size=None, key=None):
    scale = _check_py_seq(_as_jax_array(scale))
    if size is None:
      size = jnp.shape(scale)
    key = self.split_key() if key is None else _formalize_key(key)
    x = jnp.sqrt(-2. * jnp.log(jr.uniform(key, shape=_size2shape(size), minval=0, maxval=1)))
    r = x * scale
    return _return(r)

  def triangular(self, size=None, key=None):
    key = self.split_key() if key is None else _formalize_key(key)
    bernoulli_samples = jr.bernoulli(key, p=0.5, shape=_size2shape(size))
    r = 2 * bernoulli_samples - 1
    return _return(r)

  def vonmises(self, mu, kappa, size=None, key=None):
    key = self.split_key() if key is None else _formalize_key(key)
    mu = _check_py_seq(_as_jax_array(mu))
    kappa = _check_py_seq(_as_jax_array(kappa))
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(mu), jnp.shape(kappa))
    size = _size2shape(size)
    samples = _von_mises_centered(key, kappa, size)
    samples = samples + mu
    samples = (samples + jnp.pi) % (2.0 * jnp.pi) - jnp.pi
    return _return(samples)

  def weibull(self, a, size=None, key=None):
    key = self.split_key() if key is None else _formalize_key(key)
    a = _check_py_seq(_as_jax_array(a))
    if size is None:
      size = jnp.shape(a)
    else:
      if jnp.size(a) > 1:
        raise ValueError(f'"a" should be a scalar when "size" is provided. But we got {a}')
    size = _size2shape(size)
    random_uniform = jr.uniform(key=key, shape=size, minval=0, maxval=1)
    r = jnp.power(-jnp.log1p(-random_uniform), 1.0 / a)
    return _return(r)

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
    key = self.split_key() if key is None else _formalize_key(key)
    a = _check_py_seq(_as_jax_array(a))
    scale = _check_py_seq(_as_jax_array(scale))
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
    return _return(r)

  def maxwell(self, size=None, key=None):
    key = self.split_key() if key is None else _formalize_key(key)
    shape = core.canonicalize_shape(_size2shape(size)) + (3,)
    norm_rvs = jr.normal(key=key, shape=shape)
    r = jnp.linalg.norm(norm_rvs, axis=-1)
    return _return(r)

  def negative_binomial(self, n, p, size=None, key=None):
    n = _check_py_seq(_as_jax_array(n))
    p = _check_py_seq(_as_jax_array(p))
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(n), jnp.shape(p))
    size = _size2shape(size)
    logits = jnp.log(p) - jnp.log1p(-p)
    if key is None:
      keys = self.split_keys(2)
    else:
      keys = jr.split(_formalize_key(key), 2)
    rate = self.gamma(shape=n, scale=jnp.exp(-logits), size=size, key=keys[0])
    r = self.poisson(lam=rate, key=keys[1])
    return _return(r)

  def wald(self, mean, scale, size=None, key=None):
    key = self.split_key() if key is None else _formalize_key(key)
    mean = _check_py_seq(_as_jax_array(mean))
    scale = _check_py_seq(_as_jax_array(scale))
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(mean), jnp.shape(scale))
    size = _size2shape(size)
    sampled_chi2 = jnp.square(_as_jax_array(self.randn(*size)))
    sampled_uniform = _as_jax_array(self.uniform(size=size, key=key))
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
    return _return(res)

  def t(self, df, size=None, key=None):
    df = _check_py_seq(_as_jax_array(df))
    if size is None:
      size = np.shape(df)
    else:
      size = _size2shape(size)
      _check_shape("t", size, np.shape(df))
    if key is None:
      keys = self.split_keys(2)
    else:
      keys = jr.split(_formalize_key(key), 2)
    n = jr.normal(keys[0], size)
    two = _const(n, 2)
    half_df = lax.div(df, two)
    g = jr.gamma(keys[1], half_df, size)
    r = n * jnp.sqrt(half_df / g)
    return _return(r)

  def orthogonal(self, n: int, size=None, key=None):
    key = self.split_key() if key is None else _formalize_key(key)
    size = _size2shape(size)
    _check_shape("orthogonal", size)
    n = core.concrete_or_error(index, n, "The error occurred in jax.random.orthogonal()")
    z = jr.normal(key, size + (n, n))
    q, r = jnp.linalg.qr(z)
    d = jnp.diagonal(r, 0, -2, -1)
    r = q * jnp.expand_dims(d / abs(d), -2)
    return _return(r)

  def noncentral_chisquare(self, df, nonc, size=None, key=None):
    df = _check_py_seq(_as_jax_array(df))
    nonc = _check_py_seq(_as_jax_array(nonc))
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(df), jnp.shape(nonc))
    size = _size2shape(size)
    if key is None:
      keys = self.split_keys(3)
    else:
      keys = jr.split(_formalize_key(key), 3)
    i = jr.poisson(keys[0], 0.5 * nonc, shape=size)
    n = jr.normal(keys[1], shape=size) + jnp.sqrt(nonc)
    cond = jnp.greater(df, 1.0)
    df2 = jnp.where(cond, df - 1.0, df + 2.0 * i)
    chi2 = 2.0 * jr.gamma(keys[2], 0.5 * df2, shape=size)
    r = jnp.where(cond, chi2 + n * n, chi2)
    return _return(r)

  def loggamma(self, a, size=None, key=None):
    key = self.split_key() if key is None else _formalize_key(key)
    a = _check_py_seq(_as_jax_array(a))
    if size is None:
      size = jnp.shape(a)
    r = jr.loggamma(key, a, shape=_size2shape(size))
    return _return(r)

  def categorical(self, logits, axis: int = -1, size=None, key=None):
    key = self.split_key() if key is None else _formalize_key(key)
    logits = _check_py_seq(_as_jax_array(logits))
    if size is None:
      size = list(jnp.shape(logits))
      size.pop(axis)
    r = jr.categorical(key, logits, axis=axis, shape=_size2shape(size))
    return _return(r)

  def zipf(self, a, size=None, key=None):
    a = _check_py_seq(_as_jax_array(a))
    if size is None:
      size = jnp.shape(a)
    r = call(lambda x: np.random.zipf(x, size),
             a,
             result_shape=jax.ShapeDtypeStruct(size, jnp.int_))
    return _return(r)

  def power(self, a, size=None, key=None):
    a = _check_py_seq(_as_jax_array(a))
    if size is None:
      size = jnp.shape(a)
    size = _size2shape(size)
    r = call(lambda a: np.random.power(a=a, size=size),
             a, result_shape=jax.ShapeDtypeStruct(size, jnp.float_))
    return _return(r)

  def f(self, dfnum, dfden, size=None, key=None):
    dfnum = _as_jax_array(dfnum)
    dfden = _as_jax_array(dfden)
    dfnum = _check_py_seq(dfnum)
    dfden = _check_py_seq(dfden)
    if size is None:
      size = jnp.broadcast_shapes(jnp.shape(dfnum), jnp.shape(dfden))
    size = _size2shape(size)
    d = {'dfnum': dfnum, 'dfden': dfden}
    r = call(lambda x: np.random.f(dfnum=x['dfnum'],
                                   dfden=x['dfden'],
                                   size=size),
             d,
             result_shape=jax.ShapeDtypeStruct(size, jnp.float_))
    return _return(r)

  def hypergeometric(self, ngood, nbad, nsample, size=None, key=None):
    ngood = _check_py_seq(_as_jax_array(ngood))
    nbad = _check_py_seq(_as_jax_array(nbad))
    nsample = _check_py_seq(_as_jax_array(nsample))

    if size is None:
      size = lax.broadcast_shapes(jnp.shape(ngood),
                                  jnp.shape(nbad),
                                  jnp.shape(nsample))
    size = _size2shape(size)
    d = {'ngood': ngood, 'nbad': nbad, 'nsample': nsample}
    r = call(lambda x: np.random.hypergeometric(ngood=x['ngood'],
                                                nbad=x['nbad'],
                                                nsample=x['nsample'],
                                                size=size),
             d, result_shape=jax.ShapeDtypeStruct(size, jnp.int_))
    return _return(r)

  def logseries(self, p, size=None, key=None):
    p = _check_py_seq(_as_jax_array(p))
    if size is None:
      size = jnp.shape(p)
    size = _size2shape(size)
    r = call(lambda p: np.random.logseries(p=p, size=size),
             p, result_shape=jax.ShapeDtypeStruct(size, jnp.int_))
    return _return(r)

  def noncentral_f(self, dfnum, dfden, nonc, size=None, key=None):
    dfnum = _check_py_seq(_as_jax_array(dfnum))
    dfden = _check_py_seq(_as_jax_array(dfden))
    nonc = _check_py_seq(_as_jax_array(nonc))
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(dfnum),
                                  jnp.shape(dfden),
                                  jnp.shape(nonc))
    size = _size2shape(size)
    d = {'dfnum': dfnum, 'dfden': dfden, 'nonc': nonc}
    r = call(lambda x: np.random.noncentral_f(dfnum=x['dfnum'],
                                              dfden=x['dfden'],
                                              nonc=x['nonc'],
                                              size=size),
             d, result_shape=jax.ShapeDtypeStruct(size, jnp.float_))
    return _return(r)

  # PyTorch compatibility #
  # --------------------- #

  def rand_like(self, input, *, dtype=None, key=None):
    """Returns a tensor with the same size as input that is filled with random
    numbers from a uniform distribution on the interval ``[0, 1)``.

    Args:
      input:  the ``size`` of input will determine size of the output tensor.
      dtype:  the desired data type of returned Tensor. Default: if ``None``, defaults to the dtype of input.
      key: the seed or key for the random.

    Returns:
      The random data.
    """
    return self.random(shape(input), key=key).astype(dtype)

  def randn_like(self, input, *, dtype=None, key=None):
    """Returns a tensor with the same size as ``input`` that is filled with
    random numbers from a normal distribution with mean 0 and variance 1.

    Args:
      input:  the ``size`` of input will determine size of the output tensor.
      dtype:  the desired data type of returned Tensor. Default: if ``None``, defaults to the dtype of input.
      key: the seed or key for the random.

    Returns:
      The random data.
    """
    return self.randn(*shape(input), key=key).astype(dtype)

  def randint_like(self, input, low=0, high=None, *, dtype=None, key=None):
    if high is None:
      high = max(input)
    return self.randint(low, high=high, size=shape(input), dtype=dtype, key=key)


# alias
Generator = RandomState

# default random generator
__a = Array(None)
__a._value = np.random.randint(0, 10000, size=2, dtype=np.uint32)
DEFAULT = RandomState(__a)
del __a


def split_key():
  return DEFAULT.split_key()


def clone_rng(seed_or_key=None, clone: bool = True) -> RandomState:
  if seed_or_key is None:
    return DEFAULT.clone() if clone else DEFAULT
  else:
    return RandomState(seed_or_key)


def default_rng(seed_or_key=None, clone=True) -> RandomState:
  if seed_or_key is None:
    return DEFAULT.clone() if clone else DEFAULT
  else:
    return RandomState(seed_or_key)


def seed(seed: int = None):
  """Sets a new random seed.

  Parameters
  ----------
  seed: int, optional
    The random seed.
  """
  with jax.ensure_compile_time_eval():
    if seed is None:
      seed = np.random.randint(0, 100000)
    np.random.seed(seed)
  DEFAULT.seed(seed)


def rand(*dn, key=None):
  r"""Random values in a given shape.

  .. note::
      This is a convenience function for users porting code from Matlab,
      and wraps `random_sample`. That function takes a
      tuple to specify the size of the output, which is consistent with
      other NumPy functions like `numpy.zeros` and `numpy.ones`.

  Create an array of the given shape and populate it with
  random samples from a uniform distribution
  over ``[0, 1)``.

  Parameters
  ----------
  d0, d1, ..., dn : int, optional
      The dimensions of the returned array, must be non-negative.
      If no argument is given a single Python float is returned.

  Returns
  -------
  out : ndarray, shape ``(d0, d1, ..., dn)``
      Random values.

  See Also
  --------
  random

  Examples
  --------
  >>> brainpy.math.random.rand(3,2)
  array([[ 0.14022471,  0.96360618],  #random
         [ 0.37601032,  0.25528411],  #random
         [ 0.49313049,  0.94909878]]) #random
  """
  return DEFAULT.rand(*dn, key=key)


def randint(low, high=None, size=None, dtype=jnp.int_, key=None):
  r"""Return random integers from `low` (inclusive) to `high` (exclusive).

  Return random integers from the "discrete uniform" distribution of
  the specified dtype in the "half-open" interval [`low`, `high`). If
  `high` is None (the default), then results are from [0, `low`).

  Parameters
  ----------
  low : int or array-like of ints
      Lowest (signed) integers to be drawn from the distribution (unless
      ``high=None``, in which case this parameter is one above the
      *highest* such integer).
  high : int or array-like of ints, optional
      If provided, one above the largest (signed) integer to be drawn
      from the distribution (see above for behavior if ``high=None``).
      If array-like, must contain integer values
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  Default is None, in which case a
      single value is returned.
  dtype : dtype, optional
      Desired dtype of the result. Byteorder must be native.
      The default value is int.

  Returns
  -------
  out : int or ndarray of ints
      `size`-shaped array of random integers from the appropriate
      distribution, or a single such random int if `size` not provided.

  See Also
  --------
  random_integers : similar to `randint`, only for the closed
      interval [`low`, `high`], and 1 is the lowest value if `high` is
      omitted.
  Generator.integers: which should be used for new code.

  Examples
  --------
  >>> import brainpy.math as bm
  >>> bm.random.randint(2, size=10)
  array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0]) # random
  >>> bm.random.randint(1, size=10)
  array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

  Generate a 2 x 4 array of ints between 0 and 4, inclusive:

  >>> bm.random.randint(5, size=(2, 4))
  array([[4, 0, 2, 1], # random
         [3, 2, 2, 0]])

  Generate a 1 x 3 array with 3 different upper bounds

  >>> bm.random.randint(1, [3, 5, 10])
  array([2, 2, 9]) # random

  Generate a 1 by 3 array with 3 different lower bounds

  >>> bm.random.randint([1, 5, 7], 10)
  array([9, 8, 7]) # random

  Generate a 2 by 4 array using broadcasting with dtype of uint8

  >>> bm.random.randint([1, 3, 5, 7], [[10], [20]], dtype=np.uint8)
  array([[ 8,  6,  9,  7], # random
         [ 1, 16,  9, 12]], dtype=uint8)
  """

  return DEFAULT.randint(low, high=high, size=size, dtype=dtype, key=key)


def random_integers(low, high=None, size=None, key=None):
  r"""
  Random integers of type `np.int_` between `low` and `high`, inclusive.

  Return random integers of type `np.int_` from the "discrete uniform"
  distribution in the closed interval [`low`, `high`].  If `high` is
  None (the default), then results are from [1, `low`]. The `np.int_`
  type translates to the C long integer type and its precision
  is platform dependent.

  Parameters
  ----------
  low : int
      Lowest (signed) integer to be drawn from the distribution (unless
      ``high=None``, in which case this parameter is the *highest* such
      integer).
  high : int, optional
      If provided, the largest (signed) integer to be drawn from the
      distribution (see above for behavior if ``high=None``).
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  Default is None, in which case a
      single value is returned.

  Returns
  -------
  out : int or ndarray of ints
      `size`-shaped array of random integers from the appropriate
      distribution, or a single such random int if `size` not provided.

  See Also
  --------
  randint : Similar to `random_integers`, only for the half-open
      interval [`low`, `high`), and 0 is the lowest value if `high` is
      omitted.

  Notes
  -----
  To sample from N evenly spaced floating-point numbers between a and b,
  use::

    a + (b - a) * (bm.random.random_integers(N) - 1) / (N - 1.)

  Examples
  --------
  >>> import brainpy.math as bm
  >>> bm.random.random_integers(5)
  4 # random
  >>> type(bm.random.random_integers(5))
  <class 'numpy.int64'>
  >>> bm.random.random_integers(5, size=(3,2))
  array([[5, 4], # random
         [3, 3],
         [4, 5]])

  Choose five random numbers from the set of five evenly-spaced
  numbers between 0 and 2.5, inclusive (*i.e.*, from the set
  :math:`{0, 5/8, 10/8, 15/8, 20/8}`):

  >>> 2.5 * (bm.random.random_integers(5, size=(5,)) - 1) / 4.
  array([ 0.625,  1.25 ,  0.625,  0.625,  2.5  ]) # random

  Roll two six sided dice 1000 times and sum the results:

  >>> d1 = bm.random.random_integers(1, 6, 1000)
  >>> d2 = bm.random.random_integers(1, 6, 1000)
  >>> dsums = d1 + d2

  Display results as a histogram:

  >>> import matplotlib.pyplot as plt
  >>> count, bins, ignored = plt.hist(dsums, 11, density=True)
  >>> plt.show()
  """

  return DEFAULT.random_integers(low, high=high, size=size, key=key)


def randn(*dn, key=None):
  r"""
  Return a sample (or samples) from the "standard normal" distribution.

  .. note::
      This is a convenience function for users porting code from Matlab,
      and wraps `standard_normal`. That function takes a
      tuple to specify the size of the output, which is consistent with
      other NumPy functions like `numpy.zeros` and `numpy.ones`.

  .. note::
      New code should use the ``standard_normal`` method of a ``default_rng()``
      instance instead; please see the :ref:`random-quick-start`.

  If positive int_like arguments are provided, `randn` generates an array
  of shape ``(d0, d1, ..., dn)``, filled
  with random floats sampled from a univariate "normal" (Gaussian)
  distribution of mean 0 and variance 1. A single float randomly sampled
  from the distribution is returned if no argument is provided.

  Parameters
  ----------
  d0, d1, ..., dn : int, optional
      The dimensions of the returned array, must be non-negative.
      If no argument is given a single Python float is returned.

  Returns
  -------
  Z : ndarray or float
      A ``(d0, d1, ..., dn)``-shaped array of floating-point samples from
      the standard normal distribution, or a single such float if
      no parameters were supplied.

  See Also
  --------
  standard_normal : Similar, but takes a tuple as its argument.
  normal : Also accepts mu and sigma arguments.
  random.Generator.standard_normal: which should be used for new code.

  Notes
  -----
  For random samples from :math:`N(\mu, \sigma^2)`, use:

  ``sigma * bm.random.randn(...) + mu``

  Examples
  --------
  >>> import brainpy.math as bm
  >>> bm.random.randn()
  2.1923875335537315  # random

  Two-by-four array of samples from N(3, 6.25):

  >>> 3 + 2.5 * bm.random.randn(2, 4)
  array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
         [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random
  """

  return DEFAULT.randn(*dn, key=key)


def random(size=None, key=None):
  """
  Return random floats in the half-open interval [0.0, 1.0). Alias for
  `random_sample` to ease forward-porting to the new random API.
  """
  return DEFAULT.random(size, key=key)


def random_sample(size=None, key=None):
  r"""
  Return random floats in the half-open interval [0.0, 1.0).

  Results are from the "continuous uniform" distribution over the
  stated interval.  To sample :math:`Unif[a, b), b > a` multiply
  the output of `random_sample` by `(b-a)` and add `a`::

    (b - a) * random_sample() + a

  .. note::
      New code should use the ``random`` method of a ``default_rng()``
      instance instead; please see the :ref:`random-quick-start`.

  Parameters
  ----------
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  Default is None, in which case a
      single value is returned.

  Returns
  -------
  out : float or ndarray of floats
      Array of random floats of shape `size` (unless ``size=None``, in which
      case a single float is returned).

  See Also
  --------
  Generator.random: which should be used for new code.

  Examples
  --------
  >>> import brainpy.math as bm
  >>> bm.random.random_sample()
  0.47108547995356098 # random
  >>> type(bm.random.random_sample())
  <class 'float'>
  >>> bm.random.random_sample((5,))
  array([ 0.30220482,  0.86820401,  0.1654503 ,  0.11659149,  0.54323428]) # random

  Three-by-two array of random numbers from [-5, 0):

  >>> 5 * bm.random.random_sample((3, 2)) - 5
  array([[-3.99149989, -0.52338984], # random
         [-2.99091858, -0.79479508],
         [-1.23204345, -1.75224494]])
  """
  return DEFAULT.random_sample(size, key=key)


def ranf(size=None, key=None):
  """
  This is an alias of `random_sample`. See `random_sample`  for the complete
      documentation.
  """
  return DEFAULT.ranf(size, key=key)


def sample(size=None, key=None):
  """
  This is an alias of `random_sample`. See `random_sample`  for the complete
      documentation.
  """
  return DEFAULT.sample(size, key=key)


def choice(a, size=None, replace=True, p=None, key=None):
  r"""
  Generates a random sample from a given 1-D array

  Parameters
  ----------
  a : 1-D array-like or int
      If an ndarray, a random sample is generated from its elements.
      If an int, the random sample is generated as if it were ``np.arange(a)``
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  Default is None, in which case a
      single value is returned.
  replace : boolean, optional
      Whether the sample is with or without replacement. Default is True,
      meaning that a value of ``a`` can be selected multiple times.
  p : 1-D array-like, optional
      The probabilities associated with each entry in a.
      If not given, the sample assumes a uniform distribution over all
      entries in ``a``.

  Returns
  -------
  samples : single item or ndarray
      The generated random samples

  Raises
  ------
  ValueError
      If a is an int and less than zero, if a or p are not 1-dimensional,
      if a is an array-like of size 0, if p is not a vector of
      probabilities, if a and p have different lengths, or if
      replace=False and the sample size is greater than the population
      size

  See Also
  --------
  randint, shuffle, permutation
  Generator.choice: which should be used in new code

  Notes
  -----
  Setting user-specified probabilities through ``p`` uses a more general but less
  efficient sampler than the default. The general sampler produces a different sample
  than the optimized sampler even if each element of ``p`` is 1 / len(a).

  Sampling random rows from a 2-D array is not possible with this function,
  but is possible with `Generator.choice` through its ``axis`` keyword.

  Examples
  --------
  Generate a uniform random sample from np.arange(5) of size 3:

  >>> import brainpy.math as bm
  >>> bm.random.choice(5, 3)
  array([0, 3, 4]) # random
  >>> #This is equivalent to brainpy.math.random.randint(0,5,3)

  Generate a non-uniform random sample from np.arange(5) of size 3:

  >>> bm.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
  array([3, 3, 0]) # random

  Generate a uniform random sample from np.arange(5) of size 3 without
  replacement:

  >>> bm.random.choice(5, 3, replace=False)
  array([3,1,0]) # random
  >>> #This is equivalent to brainpy.math.random.permutation(np.arange(5))[:3]

  Generate a non-uniform random sample from np.arange(5) of size
  3 without replacement:

  >>> bm.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
  array([2, 3, 0]) # random

  Any of the above can be repeated with an arbitrary array-like
  instead of just integers. For instance:

  >>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
  >>> bm.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
  array(['pooh', 'pooh', 'pooh', 'Christopher', 'piglet'], # random
        dtype='<U11')
  """
  a = _as_jax_array(a)
  return DEFAULT.choice(a=a, size=size, replace=replace, p=p, key=key)


def permutation(x, axis: int = 0, independent: bool = False, key=None):
  r"""
  Randomly permute a sequence, or return a permuted range.

  If `x` is a multi-dimensional array, it is only shuffled along its
  first index.

  Parameters
  ----------
  x : int or array_like
      If `x` is an integer, randomly permute ``np.arange(x)``.
      If `x` is an array, make a copy and shuffle the elements
      randomly.

  Returns
  -------
  out : ndarray
      Permuted sequence or array range.

  See Also
  --------
  random.Generator.permutation: which should be used for new code.

  Examples
  --------
  >>> import brainpy.math as bm
  >>> bm.random.permutation(10)
  array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6]) # random

  >>> bm.random.permutation([1, 4, 9, 12, 15])
  array([15,  1,  9,  4, 12]) # random

  >>> arr = np.arange(9).reshape((3, 3))
  >>> bm.random.permutation(arr)
  array([[6, 7, 8], # random
         [0, 1, 2],
         [3, 4, 5]])
  """
  return DEFAULT.permutation(x, axis=axis, independent=independent, key=key)


def shuffle(x, axis=0, key=None):
  r"""
  Modify a sequence in-place by shuffling its contents.

  This function only shuffles the array along the first axis of a
  multi-dimensional array. The order of sub-arrays is changed but
  their contents remains the same.

  Parameters
  ----------
  x : ndarray or MutableSequence
      The array, list or mutable sequence to be shuffled.

  Returns
  -------
  None

  See Also
  --------
  random.Generator.shuffle: which should be used for new code.

  Examples
  --------
  >>> import brainpy.math as bm
  >>> arr = np.arange(10)
  >>> bm.random.shuffle(arr)
  >>> arr
  [1 7 5 2 9 4 3 6 0 8] # random

  Multi-dimensional arrays are only shuffled along the first axis:

  >>> arr = np.arange(9).reshape((3, 3))
  >>> bm.random.shuffle(arr)
  >>> arr
  array([[3, 4, 5], # random
         [6, 7, 8],
         [0, 1, 2]])
  """
  DEFAULT.shuffle(x, axis, key=key)


def beta(a, b, size=None, key=None):
  r"""
  Draw samples from a Beta distribution.

  The Beta distribution is a special case of the Dirichlet distribution,
  and is related to the Gamma distribution.  It has the probability
  distribution function

  .. math:: f(x; a,b) = \frac{1}{B(\alpha, \beta)} x^{\alpha - 1}
                                                   (1 - x)^{\beta - 1},

  where the normalization, B, is the beta function,

  .. math:: B(\alpha, \beta) = \int_0^1 t^{\alpha - 1}
                               (1 - t)^{\beta - 1} dt.

  It is often seen in Bayesian inference and order statistics.

  Parameters
  ----------
  a : float or array_like of floats
      Alpha, positive (>0).
  b : float or array_like of floats
      Beta, positive (>0).
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``a`` and ``b`` are both scalars.
      Otherwise, ``np.broadcast(a, b).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized beta distribution.

  See Also
  --------
  random.Generator.beta: which should be used for new code.
  """
  return DEFAULT.beta(a, b, size=size, key=key)


# @wraps(np.random.exponential)
def exponential(scale=None, size=None, key=None):
  return DEFAULT.exponential(scale, size, key=key)


# @wraps(np.random.gamma)
def gamma(shape, scale=None, size=None, key=None):
  return DEFAULT.gamma(shape, scale, size=size, key=key)


# @wraps(np.random.gumbel)
def gumbel(loc=None, scale=None, size=None, key=None):
  return DEFAULT.gumbel(loc, scale, size=size, key=key)


# @wraps(np.random.laplace)
def laplace(loc=None, scale=None, size=None, key=None):
  return DEFAULT.laplace(loc, scale, size, key=key)


# @wraps(np.random.logistic)
def logistic(loc=None, scale=None, size=None, key=None):
  return DEFAULT.logistic(loc, scale, size, key=key)


# @wraps(np.random.normal)
def normal(loc=None, scale=None, size=None, key=None):
  return DEFAULT.normal(loc, scale, size, key=key)


# @wraps(np.random.pareto)
def pareto(a, size=None, key=None):
  return DEFAULT.pareto(a, size, key=key)


# @wraps(np.random.poisson)
def poisson(lam=1.0, size=None, key=None):
  return DEFAULT.poisson(lam, size, key=key)


# @wraps(np.random.standard_cauchy)
def standard_cauchy(size=None, key=None):
  return DEFAULT.standard_cauchy(size, key=key)


# @wraps(np.random.standard_exponential)
def standard_exponential(size=None, key=None):
  return DEFAULT.standard_exponential(size, key=key)


# @wraps(np.random.standard_gamma)
def standard_gamma(shape, size=None, key=None):
  return DEFAULT.standard_gamma(shape, size, key=key)


# @wraps(np.random.standard_normal)
def standard_normal(size=None, key=None):
  return DEFAULT.standard_normal(size, key=key)


# @wraps(np.random.standard_t)
def standard_t(df, size=None, key=None):
  return DEFAULT.standard_t(df, size, key=key)


# @wraps(np.random.uniform)
def uniform(low=0.0, high=1.0, size=None, key=None):
  return DEFAULT.uniform(low, high, size, key=key)


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
  out : Array
    A random array with the specified dtype and shape given by ``shape`` if
    ``shape`` is not None, or else by broadcasting ``lower`` and ``upper``.
    Returns values in the open interval ``(lower, upper)``.
  """
  return DEFAULT.truncated_normal(lower, upper, size, scale, key=key)


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


# @wraps(np.random.lognormal)
def lognormal(mean=None, sigma=None, size=None, key=None):
  return DEFAULT.lognormal(mean, sigma, size, key=key)


# @wraps(np.random.binomial)
def binomial(n, p, size=None, key=None):
  return DEFAULT.binomial(n, p, size, key=key)


# @wraps(np.random.chisquare)
def chisquare(df, size=None, key=None):
  return DEFAULT.chisquare(df, size, key=key)


# @wraps(np.random.dirichlet)
def dirichlet(alpha, size=None, key=None):
  return DEFAULT.dirichlet(alpha, size, key=key)


# @wraps(np.random.geometric)
def geometric(p, size=None, key=None):
  return DEFAULT.geometric(p, size, key=key)


# @wraps(np.random.f)
def f(dfnum, dfden, size=None, key=None):
  return DEFAULT.f(dfnum, dfden, size, key=key)


# @wraps(np.random.hypergeometric)
def hypergeometric(ngood, nbad, nsample, size=None, key=None):
  return DEFAULT.hypergeometric(ngood, nbad, nsample, size, key=key)


# @wraps(np.random.logseries)
def logseries(p, size=None, key=None):
  return DEFAULT.logseries(p, size, key=key)


# @wraps(np.random.multinomial)
def multinomial(n, pvals, size=None, key=None):
  return DEFAULT.multinomial(n, pvals, size, key=key)


# @wraps(np.random.multivariate_normal)
def multivariate_normal(mean, cov, size=None, method: str = 'cholesky', key=None):
  return DEFAULT.multivariate_normal(mean, cov, size, method, key=key)


# @wraps(np.random.negative_binomial)
def negative_binomial(n, p, size=None, key=None):
  return DEFAULT.negative_binomial(n, p, size, key=key)


# @wraps(np.random.noncentral_chisquare)
def noncentral_chisquare(df, nonc, size=None, key=None):
  return DEFAULT.noncentral_chisquare(df, nonc, size, key=key)


# @wraps(np.random.noncentral_f)
def noncentral_f(dfnum, dfden, nonc, size=None, key=None):
  return DEFAULT.noncentral_f(dfnum, dfden, nonc, size, key=key)


# @wraps(np.random.power)
def power(a, size=None, key=None):
  return DEFAULT.power(a, size, key=key)


# @wraps(np.random.rayleigh)
def rayleigh(scale=1.0, size=None, key=None):
  return DEFAULT.rayleigh(scale, size, key=key)


# @wraps(np.random.triangular)
def triangular(size=None, key=None):
  return DEFAULT.triangular(size, key=key)


# @wraps(np.random.vonmises)
def vonmises(mu, kappa, size=None, key=None):
  return DEFAULT.vonmises(mu, kappa, size, key=key)


# @wraps(np.random.wald)
def wald(mean, scale, size=None, key=None):
  return DEFAULT.wald(mean, scale, size, key=key)


def weibull(a, size=None, key=None):
  r"""
  Draw samples from a Weibull distribution.
    
  Draw samples from a 1-parameter Weibull distribution with the given
  shape parameter `a`.

  .. math:: X = (-ln(U))^{1/a}

  Here, U is drawn from the uniform distribution over (0,1].

  The more common 2-parameter Weibull, including a scale parameter
  :math:`\lambda` is just :math:`X = \lambda(-ln(U))^{1/a}`.

  .. note::
      New code should use the ``weibull`` method of a ``default_rng()``
      instance instead; please see the :ref:`random-quick-start`.

  Parameters
  ----------
  a : float or array_like of floats
      Shape parameter of the distribution.  Must be nonnegative.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``a`` is a scalar.  Otherwise,
      ``np.array(a).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized Weibull distribution.

  See Also
  --------
  scipy.stats.weibull_max
  scipy.stats.weibull_min
  scipy.stats.genextreme
  gumbel
  random.Generator.weibull: which should be used for new code.

  Notes
  -----
  The Weibull (or Type III asymptotic extreme value distribution
  for smallest values, SEV Type III, or Rosin-Rammler
  distribution) is one of a class of Generalized Extreme Value
  (GEV) distributions used in modeling extreme value problems.
  This class includes the Gumbel and Frechet distributions.

  The probability density for the Weibull distribution is

  .. math:: p(x) = \frac{a}
                   {\lambda}(\frac{x}{\lambda})^{a-1}e^{-(x/\lambda)^a},

  where :math:`a` is the shape and :math:`\lambda` the scale.

  The function has its peak (the mode) at
  :math:`\lambda(\frac{a-1}{a})^{1/a}`.

  When ``a = 1``, the Weibull distribution reduces to the exponential
  distribution.

  References
  ----------
  .. [1] Waloddi Weibull, Royal Technical University, Stockholm,
         1939 "A Statistical Theory Of The Strength Of Materials",
         Ingeniorsvetenskapsakademiens Handlingar Nr 151, 1939,
         Generalstabens Litografiska Anstalts Forlag, Stockholm.
  .. [2] Waloddi Weibull, "A Statistical Distribution Function of
         Wide Applicability", Journal Of Applied Mechanics ASME Paper
         1951.
  .. [3] Wikipedia, "Weibull distribution",
         https://en.wikipedia.org/wiki/Weibull_distribution

  Examples
  --------
  Draw samples from the distribution:

  >>> a = 5. # shape
  >>> s = brainpy.math.random.weibull(a, 1000)

  Display the histogram of the samples, along with
  the probability density function:

  >>> import matplotlib.pyplot as plt
  >>> x = np.arange(1,100.)/50.
  >>> def weib(x,n,a):
  ...     return (a / n) * (x / n)**(a - 1) * np.exp(-(x / n)**a)

  >>> count, bins, ignored = plt.hist(brainpy.math.random.weibull(5.,1000))
  >>> x = np.arange(1,100.)/50.
  >>> scale = count.max()/weib(x, 1., 5.).max()
  >>> plt.plot(x, weib(x, 1., 5.)*scale)
  >>> plt.show()

  """
  return DEFAULT.weibull(a, size, key=key)


def weibull_min(a, scale=None, size=None, key=None):
  """Sample from a Weibull distribution.

  The scipy counterpart is `scipy.stats.weibull_min`.

  Args:
    scale: The scale parameter of the distribution.
    concentration: The concentration parameter of the distribution.
    shape: The shape added to the parameters loc and scale broadcastable shape.
    dtype: The type used for samples.
    key: a PRNG key or a seed.

  Returns:
    A jnp.array of samples.

  """
  return DEFAULT.weibull_min(a, scale, size, key=key)


def zipf(a, size=None, key=None):
  r"""
  Draw samples from a Zipf distribution.

  Samples are drawn from a Zipf distribution with specified parameter
  `a` > 1.

  The Zipf distribution (also known as the zeta distribution) is a
  discrete probability distribution that satisfies Zipf's law: the
  frequency of an item is inversely proportional to its rank in a
  frequency table.

  .. note::
      New code should use the ``zipf`` method of a ``default_rng()``
      instance instead; please see the :ref:`random-quick-start`.

  Parameters
  ----------
  a : float or array_like of floats
      Distribution parameter. Must be greater than 1.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``a`` is a scalar. Otherwise,
      ``np.array(a).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized Zipf distribution.

  See Also
  --------
  scipy.stats.zipf : probability density function, distribution, or
      cumulative density function, etc.
  random.Generator.zipf: which should be used for new code.

  Notes
  -----
  The probability density for the Zipf distribution is

  .. math:: p(k) = \frac{k^{-a}}{\zeta(a)},

  for integers :math:`k \geq 1`, where :math:`\zeta` is the Riemann Zeta
  function.

  It is named for the American linguist George Kingsley Zipf, who noted
  that the frequency of any word in a sample of a language is inversely
  proportional to its rank in the frequency table.

  References
  ----------
  .. [1] Zipf, G. K., "Selected Studies of the Principle of Relative
         Frequency in Language," Cambridge, MA: Harvard Univ. Press,
         1932.

  Examples
  --------
  Draw samples from the distribution:

  >>> a = 4.0
  >>> n = 20000
  >>> s = brainpy.math.random.zipf(a, n)

  Display the histogram of the samples, along with
  the expected histogram based on the probability
  density function:

  >>> import matplotlib.pyplot as plt
  >>> from scipy.special import zeta  # doctest: +SKIP

  `bincount` provides a fast histogram for small integers.

  >>> count = np.bincount(s)
  >>> k = np.arange(1, s.max() + 1)

  >>> plt.bar(k, count[1:], alpha=0.5, label='sample count')
  >>> plt.plot(k, n*(k**-a)/zeta(a), 'k.-', alpha=0.5,
  ...          label='expected count')   # doctest: +SKIP
  >>> plt.semilogy()
  >>> plt.grid(alpha=0.4)
  >>> plt.legend()
  >>> plt.title(f'Zipf sample, a={a}, size={n}')
  >>> plt.show()
  """
  return DEFAULT.zipf(a, size, key=key)


def maxwell(size=None, key=None):
  """Sample from a one sided Maxwell distribution.

  The scipy counterpart is `scipy.stats.maxwell`.

  Args:
    key: a PRNG key.
    size: The shape of the returned samples.
    dtype: The type used for samples.

  Returns:
    A jnp.array of samples, of shape `shape`.

  """
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
  out: Array
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


def categorical(logits, axis: int = -1, size=None, key=None):
  """Sample random values from categorical distributions.

  Args:
    logits: Unnormalized log probabilities of the categorical distribution(s) to sample from,
      so that `softmax(logits, axis)` gives the corresponding probabilities.
    axis: Axis along which logits belong to the same categorical distribution.
    shape: Optional, a tuple of nonnegative integers representing the result shape.
      Must be broadcast-compatible with ``np.delete(logits.shape, axis)``.
      The default (None) produces a result shape equal to ``np.delete(logits.shape, axis)``.
    key: a PRNG key used as the random key.

  Returns:
    A random array with int dtype and shape given by ``shape`` if ``shape``
    is not None, or else ``np.delete(logits.shape, axis)``.
  """
  return DEFAULT.categorical(logits, axis, size, key=key)


def rand_like(input, *, dtype=None, key=None):
  """Similar to ``rand_like`` in torch. 
  
  Returns a tensor with the same size as input that is filled with random
  numbers from a uniform distribution on the interval ``[0, 1)``.

  Args:
    input:  the ``size`` of input will determine size of the output tensor.
    dtype:  the desired data type of returned Tensor. Default: if ``None``, defaults to the dtype of input.
    key: the seed or key for the random.

  Returns:
    The random data.
  """
  return DEFAULT.rand_like(input, dtype=dtype, key=key)


def randn_like(input, *, dtype=None, key=None):
  """Similar to ``randn_like`` in torch. 
  
  Returns a tensor with the same size as ``input`` that is filled with
  random numbers from a normal distribution with mean 0 and variance 1.

  Args:
    input:  the ``size`` of input will determine size of the output tensor.
    dtype:  the desired data type of returned Tensor. Default: if ``None``, defaults to the dtype of input.
    key: the seed or key for the random.

  Returns:
    The random data.
  """
  return DEFAULT.randn_like(input, dtype=dtype, key=key)


def randint_like(input, low=0, high=None, *, dtype=None, key=None):
  """Similar to ``randint_like`` in torch. 
  
  Returns a tensor with the same shape as Tensor ``input`` filled with
  random integers generated uniformly between ``low`` (inclusive) and ``high`` (exclusive).

  Args:
    input:  the ``size`` of input will determine size of the output tensor.
    low: Lowest integer to be drawn from the distribution. Default: 0.
    high: One above the highest integer to be drawn from the distribution.
    dtype: the desired data type of returned Tensor. Default: if ``None``, defaults to the dtype of input.
    key: the seed or key for the random.

  Returns:
    The random data.
  """
  return DEFAULT.randint_like(input=input, low=low, high=high, dtype=dtype, key=key)


for __k in dir(RandomState):
  __t = getattr(RandomState, __k)
  if not __k.startswith('__') and callable(__t) and (not __t.__doc__):
    __r = globals().get(__k, None)
    if __r is not None and callable(__r):
      __t.__doc__ = __r.__doc__
