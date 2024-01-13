# -*- coding: utf-8 -*-

import warnings
from collections import namedtuple
from functools import partial
from operator import index
from typing import Optional, Union, Sequence

import jax
import numpy as np
from jax import lax, jit, vmap, numpy as jnp, random as jr, core, dtypes
from jax._src.array import ArrayImpl
from jax.experimental.host_callback import call
from jax.tree_util import register_pytree_node_class

from brainpy.check import jit_error_checking, jit_error_checking_no_args
from .compat_numpy import shape
from .environment import get_int
from .ndarray import Array, _return
from .object_transform.variables import Variable

__all__ = [
  'RandomState', 'Generator', 'DEFAULT',

  'seed', 'default_rng', 'split_key', 'split_keys',

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

JAX_RAND_KEY = jax.Array


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
  elif isinstance(size, (tuple, list)):
    return tuple(size)
  else:
    return (size,)


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
      ready_to_trace: bool = True,
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
    super(RandomState, self).__init__(key, ready_to_trace=ready_to_trace)

  def __repr__(self) -> str:
    print_code = repr(self.value)
    i = print_code.index('(')
    name = self.__class__.__name__
    return f'{name}(key={print_code[i:]})'

  @property
  def value(self):
    if isinstance(self._value, ArrayImpl):
      if self._value.is_deleted():
        self.seed()
    self._append_to_stack()
    return self._value

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

  def rand(self, *dn, key: Optional[Union[int, JAX_RAND_KEY]] = None):
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.uniform(key, shape=dn, minval=0., maxval=1.)
    return _return(r)

  def randint(self,
              low,
              high=None,
              size: Optional[Union[int, Sequence[int]]] = None,
              dtype=int, key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def random_integers(self,
                      low,
                      high=None,
                      size: Optional[Union[int, Sequence[int]]] = None,
                      key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def randn(self, *dn, key: Optional[Union[int, JAX_RAND_KEY]] = None):
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.normal(key, shape=dn)
    return _return(r)

  def random(self,
             size: Optional[Union[int, Sequence[int]]] = None,
             key: Optional[Union[int, JAX_RAND_KEY]] = None):
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.uniform(key, shape=_size2shape(size), minval=0., maxval=1.)
    return _return(r)

  def random_sample(self,
                    size: Optional[Union[int, Sequence[int]]] = None,
                    key: Optional[Union[int, JAX_RAND_KEY]] = None):
    r = self.random(size=size, key=key)
    return _return(r)

  def ranf(self, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
    r = self.random(size=size, key=key)
    return _return(r)

  def sample(self, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
    r = self.random(size=size, key=key)
    return _return(r)

  def choice(self, a, size: Optional[Union[int, Sequence[int]]] = None, replace=True, p=None,
             key: Optional[Union[int, JAX_RAND_KEY]] = None):
    a = _as_jax_array(a)
    p = _as_jax_array(p)
    a = _check_py_seq(a)
    p = _check_py_seq(p)
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.choice(key, a=a, shape=_size2shape(size), replace=replace, p=p)
    return _return(r)

  def permutation(self, x, axis: int = 0, independent: bool = False, key: Optional[Union[int, JAX_RAND_KEY]] = None):
    x = x.value if isinstance(x, Array) else x
    x = _check_py_seq(x)
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.permutation(key, x, axis=axis, independent=independent)
    return _return(r)

  def shuffle(self, x, axis=0, key: Optional[Union[int, JAX_RAND_KEY]] = None):
    if not isinstance(x, Array):
      raise TypeError('This numpy operator needs in-place updating, therefore '
                      'inputs should be brainpy Array.')
    key = self.split_key() if key is None else _formalize_key(key)
    x.value = jr.permutation(key, x.value, axis=axis)

  def beta(self, a, b,
           size: Optional[Union[int, Sequence[int]]] = None,
           key: Optional[Union[int, JAX_RAND_KEY]] = None):
    a = a.value if isinstance(a, Array) else a
    b = b.value if isinstance(b, Array) else b
    a = _check_py_seq(a)
    b = _check_py_seq(b)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(a), jnp.shape(b))
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.beta(key, a=a, b=b, shape=_size2shape(size))
    return _return(r)

  def exponential(self, scale=None,
                  size: Optional[Union[int, Sequence[int]]] = None,
                  key: Optional[Union[int, JAX_RAND_KEY]] = None):
    scale = _as_jax_array(scale)
    scale = _check_py_seq(scale)
    if size is None:
      size = jnp.shape(scale)
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.exponential(key, shape=_size2shape(size))
    if scale is not None:
      r = r / scale
    return _return(r)

  def gamma(self, shape, scale=None,
            size: Optional[Union[int, Sequence[int]]] = None,
            key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def gumbel(self, loc=None, scale=None,
             size: Optional[Union[int, Sequence[int]]] = None,
             key: Optional[Union[int, JAX_RAND_KEY]] = None):
    loc = _as_jax_array(loc)
    scale = _as_jax_array(scale)
    loc = _check_py_seq(loc)
    scale = _check_py_seq(scale)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
    key = self.split_key() if key is None else _formalize_key(key)
    r = _loc_scale(loc, scale, jr.gumbel(key, shape=_size2shape(size)))
    return _return(r)

  def laplace(self, loc=None, scale=None,
              size: Optional[Union[int, Sequence[int]]] = None,
              key: Optional[Union[int, JAX_RAND_KEY]] = None):
    loc = _as_jax_array(loc)
    scale = _as_jax_array(scale)
    loc = _check_py_seq(loc)
    scale = _check_py_seq(scale)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
    key = self.split_key() if key is None else _formalize_key(key)
    r = _loc_scale(loc, scale, jr.laplace(key, shape=_size2shape(size)))
    return _return(r)

  def logistic(self, loc=None, scale=None,
               size: Optional[Union[int, Sequence[int]]] = None,
               key: Optional[Union[int, JAX_RAND_KEY]] = None):
    loc = _as_jax_array(loc)
    scale = _as_jax_array(scale)
    loc = _check_py_seq(loc)
    scale = _check_py_seq(scale)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(loc), jnp.shape(scale))
    key = self.split_key() if key is None else _formalize_key(key)
    r = _loc_scale(loc, scale, jr.logistic(key, shape=_size2shape(size)))
    return _return(r)

  def normal(self, loc=None, scale=None,
             size: Optional[Union[int, Sequence[int]]] = None,
             key: Optional[Union[int, JAX_RAND_KEY]] = None):
    loc = _as_jax_array(loc)
    scale = _as_jax_array(scale)
    loc = _check_py_seq(loc)
    scale = _check_py_seq(scale)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(scale), jnp.shape(loc))
    key = self.split_key() if key is None else _formalize_key(key)
    r = _loc_scale(loc, scale, jr.normal(key, shape=_size2shape(size)))
    return _return(r)

  def pareto(self, a,
             size: Optional[Union[int, Sequence[int]]] = None,
             key: Optional[Union[int, JAX_RAND_KEY]] = None):
    a = _as_jax_array(a)
    a = _check_py_seq(a)
    if size is None:
      size = jnp.shape(a)
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.pareto(key, b=a, shape=_size2shape(size))
    return _return(r)

  def poisson(self, lam=1.0,
              size: Optional[Union[int, Sequence[int]]] = None,
              key: Optional[Union[int, JAX_RAND_KEY]] = None):
    lam = _check_py_seq(_as_jax_array(lam))
    if size is None:
      size = jnp.shape(lam)
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.poisson(key, lam=lam, shape=_size2shape(size))
    return _return(r)

  def standard_cauchy(self,
                      size: Optional[Union[int, Sequence[int]]] = None,
                      key: Optional[Union[int, JAX_RAND_KEY]] = None):
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.cauchy(key, shape=_size2shape(size))
    return _return(r)

  def standard_exponential(self,
                           size: Optional[Union[int, Sequence[int]]] = None,
                           key: Optional[Union[int, JAX_RAND_KEY]] = None):
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.exponential(key, shape=_size2shape(size))
    return _return(r)

  def standard_gamma(self,
                     shape,
                     size: Optional[Union[int, Sequence[int]]] = None,
                     key: Optional[Union[int, JAX_RAND_KEY]] = None):
    shape = _as_jax_array(shape)
    shape = _check_py_seq(shape)
    if size is None:
      size = jnp.shape(shape)
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.gamma(key, a=shape, shape=_size2shape(size))
    return _return(r)

  def standard_normal(self,
                      size: Optional[Union[int, Sequence[int]]] = None,
                      key: Optional[Union[int, JAX_RAND_KEY]] = None):
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.normal(key, shape=_size2shape(size))
    return _return(r)

  def standard_t(self, df,
                 size: Optional[Union[int, Sequence[int]]] = None,
                 key: Optional[Union[int, JAX_RAND_KEY]] = None):
    df = _as_jax_array(df)
    df = _check_py_seq(df)
    if size is None:
      size = jnp.shape(size)
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.t(key, df=df, shape=_size2shape(size))
    return _return(r)

  def uniform(self, low=0.0, high=1.0,
              size: Optional[Union[int, Sequence[int]]] = None,
              key: Optional[Union[int, JAX_RAND_KEY]] = None):
    low = _as_jax_array(low)
    high = _as_jax_array(high)
    low = _check_py_seq(low)
    high = _check_py_seq(high)
    if size is None:
      size = lax.broadcast_shapes(jnp.shape(low), jnp.shape(high))
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.uniform(key, shape=_size2shape(size), minval=low, maxval=high)
    return _return(r)

  def __norm_cdf(self, x, sqrt2, dtype):
    # Computes standard normal cumulative distribution function
    return (np.asarray(1., dtype) + lax.erf(x / sqrt2)) / np.asarray(2., dtype)

  def truncated_normal(self,
                       lower,
                       upper,
                       size: Optional[Union[int, Sequence[int]]] = None,
                       loc=0.,
                       scale=1.,
                       dtype=float,
                       key: Optional[Union[int, JAX_RAND_KEY]] = None):
    lower = _check_py_seq(_as_jax_array(lower))
    upper = _check_py_seq(_as_jax_array(upper))
    loc = _check_py_seq(_as_jax_array(loc))
    scale = _check_py_seq(_as_jax_array(scale))

    lower = lax.convert_element_type(lower, dtype)
    upper = lax.convert_element_type(upper, dtype)
    loc = lax.convert_element_type(loc, dtype)
    scale = lax.convert_element_type(scale, dtype)

    jit_error_checking_no_args(
      jnp.any(jnp.logical_or(loc < lower - 2 * scale, loc > upper + 2 * scale)),
      ValueError("mean is more than 2 std from [lower, upper] in truncated_normal. "
                 "The distribution of values may be incorrect.")
    )

    if size is None:
      size = lax.broadcast_shapes(jnp.shape(lower),
                                  jnp.shape(upper),
                                  jnp.shape(loc),
                                  jnp.shape(scale))

    # Values are generated by using a truncated uniform distribution and
    # then using the inverse CDF for the normal distribution.
    # Get upper and lower cdf values
    sqrt2 = np.array(np.sqrt(2), dtype)
    l = self.__norm_cdf((lower - loc) / scale, sqrt2, dtype)
    u = self.__norm_cdf((upper - loc) / scale, sqrt2, dtype)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    key = self.split_key() if key is None else _formalize_key(key)
    out = jr.uniform(key, size, dtype,
                     minval=lax.nextafter(2 * l - 1, np.array(np.inf, dtype=dtype)),
                     maxval=lax.nextafter(2 * u - 1, np.array(-np.inf, dtype=dtype)))

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    out = lax.erf_inv(out)

    # Transform to proper mean, std
    out = out * scale * sqrt2 + loc

    # Clamp to ensure it's in the proper range
    out = jnp.clip(out,
                   lax.nextafter(lax.stop_gradient(lower), np.array(np.inf, dtype=dtype)),
                   lax.nextafter(lax.stop_gradient(upper), np.array(-np.inf, dtype=dtype)))
    return _return(out)

  def _check_p(self, p):
    raise ValueError(f'Parameter p should be within [0, 1], but we got {p}')

  def bernoulli(self, p, size: Optional[Union[int, Sequence[int]]] = None,
                key: Optional[Union[int, JAX_RAND_KEY]] = None):
    p = _check_py_seq(_as_jax_array(p))
    jit_error_checking(jnp.any(jnp.logical_and(p < 0, p > 1)), self._check_p, p)
    if size is None:
      size = jnp.shape(p)
    key = self.split_key() if key is None else _formalize_key(key)
    r = jr.bernoulli(key, p=p, shape=_size2shape(size))
    return _return(r)

  def lognormal(self, mean=None, sigma=None, size: Optional[Union[int, Sequence[int]]] = None,
                key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def binomial(self, n, p, size: Optional[Union[int, Sequence[int]]] = None,
               key: Optional[Union[int, JAX_RAND_KEY]] = None):
    n = _check_py_seq(n.value if isinstance(n, Array) else n)
    p = _check_py_seq(p.value if isinstance(p, Array) else p)
    jit_error_checking(jnp.any(jnp.logical_and(p < 0, p > 1)), self._check_p, p)
    if size is None:
      size = jnp.broadcast_shapes(jnp.shape(n), jnp.shape(p))
    key = self.split_key() if key is None else _formalize_key(key)
    r = _binomial(key, p, n, shape=_size2shape(size))
    return _return(r)

  def chisquare(self, df, size: Optional[Union[int, Sequence[int]]] = None,
                key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def dirichlet(self, alpha, size: Optional[Union[int, Sequence[int]]] = None,
                key: Optional[Union[int, JAX_RAND_KEY]] = None):
    key = self.split_key() if key is None else _formalize_key(key)
    alpha = _check_py_seq(_as_jax_array(alpha))
    r = jr.dirichlet(key, alpha=alpha, shape=_size2shape(size))
    return _return(r)

  def geometric(self, p, size: Optional[Union[int, Sequence[int]]] = None,
                key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def multinomial(self, n, pvals, size: Optional[Union[int, Sequence[int]]] = None,
                  key: Optional[Union[int, JAX_RAND_KEY]] = None):
    key = self.split_key() if key is None else _formalize_key(key)
    n = _check_py_seq(_as_jax_array(n))
    pvals = _check_py_seq(_as_jax_array(pvals))
    jit_error_checking(jnp.sum(pvals[:-1]) > 1., self._check_p2, pvals)
    if isinstance(n, jax.core.Tracer):
      raise ValueError("The total count parameter `n` should not be a jax abstract array.")
    size = _size2shape(size)
    n_max = int(np.max(jax.device_get(n)))
    batch_shape = lax.broadcast_shapes(jnp.shape(pvals)[:-1], jnp.shape(n))
    r = _multinomial(key, pvals, n, n_max, batch_shape + size)
    return _return(r)

  def multivariate_normal(self, mean, cov, size: Optional[Union[int, Sequence[int]]] = None, method: str = 'cholesky',
                          key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def rayleigh(self, scale=1.0, size: Optional[Union[int, Sequence[int]]] = None,
               key: Optional[Union[int, JAX_RAND_KEY]] = None):
    scale = _check_py_seq(_as_jax_array(scale))
    if size is None:
      size = jnp.shape(scale)
    key = self.split_key() if key is None else _formalize_key(key)
    x = jnp.sqrt(-2. * jnp.log(jr.uniform(key, shape=_size2shape(size), minval=0, maxval=1)))
    r = x * scale
    return _return(r)

  def triangular(self, size: Optional[Union[int, Sequence[int]]] = None,
                 key: Optional[Union[int, JAX_RAND_KEY]] = None):
    key = self.split_key() if key is None else _formalize_key(key)
    bernoulli_samples = jr.bernoulli(key, p=0.5, shape=_size2shape(size))
    r = 2 * bernoulli_samples - 1
    return _return(r)

  def vonmises(self, mu, kappa, size: Optional[Union[int, Sequence[int]]] = None,
               key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def weibull(self, a, size: Optional[Union[int, Sequence[int]]] = None,
              key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def weibull_min(self, a, scale=None, size: Optional[Union[int, Sequence[int]]] = None,
                  key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def maxwell(self, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
    key = self.split_key() if key is None else _formalize_key(key)
    shape = core.canonicalize_shape(_size2shape(size)) + (3,)
    norm_rvs = jr.normal(key=key, shape=shape)
    r = jnp.linalg.norm(norm_rvs, axis=-1)
    return _return(r)

  def negative_binomial(self, n, p, size: Optional[Union[int, Sequence[int]]] = None,
                        key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def wald(self, mean, scale, size: Optional[Union[int, Sequence[int]]] = None,
           key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def t(self, df, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def orthogonal(self, n: int, size: Optional[Union[int, Sequence[int]]] = None,
                 key: Optional[Union[int, JAX_RAND_KEY]] = None):
    key = self.split_key() if key is None else _formalize_key(key)
    size = _size2shape(size)
    _check_shape("orthogonal", size)
    n = core.concrete_or_error(index, n, "The error occurred in jax.random.orthogonal()")
    z = jr.normal(key, size + (n, n))
    q, r = jnp.linalg.qr(z)
    d = jnp.diagonal(r, 0, -2, -1)
    r = q * jnp.expand_dims(d / abs(d), -2)
    return _return(r)

  def noncentral_chisquare(self, df, nonc, size: Optional[Union[int, Sequence[int]]] = None,
                           key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def loggamma(self, a, size: Optional[Union[int, Sequence[int]]] = None,
               key: Optional[Union[int, JAX_RAND_KEY]] = None):
    key = self.split_key() if key is None else _formalize_key(key)
    a = _check_py_seq(_as_jax_array(a))
    if size is None:
      size = jnp.shape(a)
    r = jr.loggamma(key, a, shape=_size2shape(size))
    return _return(r)

  def categorical(self, logits, axis: int = -1, size: Optional[Union[int, Sequence[int]]] = None,
                  key: Optional[Union[int, JAX_RAND_KEY]] = None):
    key = self.split_key() if key is None else _formalize_key(key)
    logits = _check_py_seq(_as_jax_array(logits))
    if size is None:
      size = list(jnp.shape(logits))
      size.pop(axis)
    r = jr.categorical(key, logits, axis=axis, shape=_size2shape(size))
    return _return(r)

  def zipf(self, a, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
    a = _check_py_seq(_as_jax_array(a))
    if size is None:
      size = jnp.shape(a)
    r = call(lambda x: np.random.zipf(x, size),
             a,
             result_shape=jax.ShapeDtypeStruct(size, jnp.int_))
    return _return(r)

  def power(self, a, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
    a = _check_py_seq(_as_jax_array(a))
    if size is None:
      size = jnp.shape(a)
    size = _size2shape(size)
    r = call(lambda a: np.random.power(a=a, size=size),
             a, result_shape=jax.ShapeDtypeStruct(size, jnp.float_))
    return _return(r)

  def f(self, dfnum, dfden, size: Optional[Union[int, Sequence[int]]] = None,
        key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def hypergeometric(self, ngood, nbad, nsample, size: Optional[Union[int, Sequence[int]]] = None,
                     key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def logseries(self, p, size: Optional[Union[int, Sequence[int]]] = None,
                key: Optional[Union[int, JAX_RAND_KEY]] = None):
    p = _check_py_seq(_as_jax_array(p))
    if size is None:
      size = jnp.shape(p)
    size = _size2shape(size)
    r = call(lambda p: np.random.logseries(p=p, size=size),
             p, result_shape=jax.ShapeDtypeStruct(size, jnp.int_))
    return _return(r)

  def noncentral_f(self, dfnum, dfden, nonc, size: Optional[Union[int, Sequence[int]]] = None,
                   key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def rand_like(self, input, *, dtype=None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def randn_like(self, input, *, dtype=None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
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

  def randint_like(self, input, low=0, high=None, *, dtype=None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
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
  """Create a new seed from the current seed.

  This function is useful for the consistency with JAX's random paradigm."""
  return DEFAULT.split_key()


def split_keys(n):
  """Create multiple seeds from the current seed. This is used
  internally by `pmap` and `vmap` to ensure that random numbers
  are different in parallel threads.

  .. versionadded:: 2.4.5

  Parameters
  ----------
  n : int
    The number of seeds to generate.
  """
  return DEFAULT.split_keys(n)


def clone_rng(seed_or_key=None, clone: bool = True) -> RandomState:
  """Clone the random state according to the given setting.

  Args:
    seed_or_key: The seed (an integer) or the random key.
    clone: Bool. Whether clone the default random state.

  Returns:
    The random state.
  """
  if seed_or_key is None:
    return DEFAULT.clone() if clone else DEFAULT
  else:
    return RandomState(seed_or_key)


def default_rng(seed_or_key=None, clone: bool = True) -> RandomState:
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


def rand(*dn, key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def randint(low, high=None, size: Optional[Union[int, Sequence[int]]] = None, dtype=int,
            key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def random_integers(low,
                    high=None,
                    size: Optional[Union[int, Sequence[int]]] = None,
                    key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def randn(*dn, key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def random(size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Return random floats in the half-open interval [0.0, 1.0). Alias for
  `random_sample` to ease forward-porting to the new random API.
  """
  return DEFAULT.random(size, key=key)


def random_sample(size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def ranf(size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  This is an alias of `random_sample`. See `random_sample`  for the complete
  documentation.
  """
  return DEFAULT.ranf(size, key=key)


def sample(size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
  """
  This is an alias of `random_sample`. See `random_sample`  for the complete
  documentation.
  """
  return DEFAULT.sample(size, key=key)


def choice(a, size: Optional[Union[int, Sequence[int]]] = None, replace=True, p=None,
           key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def permutation(x,
                axis: int = 0,
                independent: bool = False,
                key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def shuffle(x, axis=0, key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def beta(a, b, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
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
  """
  return DEFAULT.beta(a, b, size=size, key=key)


def exponential(scale=None, size: Optional[Union[int, Sequence[int]]] = None,
                key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from an exponential distribution.

  Its probability density function is

  .. math:: f(x; \frac{1}{\beta}) = \frac{1}{\beta} \exp(-\frac{x}{\beta}),

  for ``x > 0`` and 0 elsewhere. :math:`\beta` is the scale parameter,
  which is the inverse of the rate parameter :math:`\lambda = 1/\beta`.
  The rate parameter is an alternative, widely used parameterization
  of the exponential distribution [3]_.

  The exponential distribution is a continuous analogue of the
  geometric distribution.  It describes many common situations, such as
  the size of raindrops measured over many rainstorms [1]_, or the time
  between page requests to Wikipedia [2]_.

  Parameters
  ----------
  scale : float or array_like of floats
      The scale parameter, :math:`\beta = 1/\lambda`. Must be
      non-negative.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``scale`` is a scalar.  Otherwise,
      ``np.array(scale).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized exponential distribution.

  References
  ----------
  .. [1] Peyton Z. Peebles Jr., "Probability, Random Variables and
         Random Signal Principles", 4th ed, 2001, p. 57.
  .. [2] Wikipedia, "Poisson process",
         https://en.wikipedia.org/wiki/Poisson_process
  .. [3] Wikipedia, "Exponential distribution",
         https://en.wikipedia.org/wiki/Exponential_distribution
  """
  return DEFAULT.exponential(scale, size, key=key)


def gamma(shape, scale=None, size: Optional[Union[int, Sequence[int]]] = None,
          key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a Gamma distribution.

  Samples are drawn from a Gamma distribution with specified parameters,
  `shape` (sometimes designated "k") and `scale` (sometimes designated
  "theta"), where both parameters are > 0.

  Parameters
  ----------
  shape : float or array_like of floats
      The shape of the gamma distribution. Must be non-negative.
  scale : float or array_like of floats, optional
      The scale of the gamma distribution. Must be non-negative.
      Default is equal to 1.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``shape`` and ``scale`` are both scalars.
      Otherwise, ``np.broadcast(shape, scale).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized gamma distribution.


  Notes
  -----
  The probability density for the Gamma distribution is

  .. math:: p(x) = x^{k-1}\frac{e^{-x/\theta}}{\theta^k\Gamma(k)},

  where :math:`k` is the shape and :math:`\theta` the scale,
  and :math:`\Gamma` is the Gamma function.

  The Gamma distribution is often used to model the times to failure of
  electronic components, and arises naturally in processes for which the
  waiting times between Poisson distributed events are relevant.

  References
  ----------
  .. [1] Weisstein, Eric W. "Gamma Distribution." From MathWorld--A
         Wolfram Web Resource.
         http://mathworld.wolfram.com/GammaDistribution.html
  .. [2] Wikipedia, "Gamma distribution",
         https://en.wikipedia.org/wiki/Gamma_distribution

  """
  return DEFAULT.gamma(shape, scale, size=size, key=key)


def gumbel(loc=None, scale=None, size: Optional[Union[int, Sequence[int]]] = None,
           key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a Gumbel distribution.

  Draw samples from a Gumbel distribution with specified location and
  scale.  For more information on the Gumbel distribution, see
  Notes and References below.

  Parameters
  ----------
  loc : float or array_like of floats, optional
      The location of the mode of the distribution. Default is 0.
  scale : float or array_like of floats, optional
      The scale parameter of the distribution. Default is 1. Must be non-
      negative.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``loc`` and ``scale`` are both scalars.
      Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized Gumbel distribution.

  Notes
  -----
  The Gumbel (or Smallest Extreme Value (SEV) or the Smallest Extreme
  Value Type I) distribution is one of a class of Generalized Extreme
  Value (GEV) distributions used in modeling extreme value problems.
  The Gumbel is a special case of the Extreme Value Type I distribution
  for maximums from distributions with "exponential-like" tails.

  The probability density for the Gumbel distribution is

  .. math:: p(x) = \frac{e^{-(x - \mu)/ \beta}}{\beta} e^{ -e^{-(x - \mu)/
            \beta}},

  where :math:`\mu` is the mode, a location parameter, and
  :math:`\beta` is the scale parameter.

  The Gumbel (named for German mathematician Emil Julius Gumbel) was used
  very early in the hydrology literature, for modeling the occurrence of
  flood events. It is also used for modeling maximum wind speed and
  rainfall rates.  It is a "fat-tailed" distribution - the probability of
  an event in the tail of the distribution is larger than if one used a
  Gaussian, hence the surprisingly frequent occurrence of 100-year
  floods. Floods were initially modeled as a Gaussian process, which
  underestimated the frequency of extreme events.

  It is one of a class of extreme value distributions, the Generalized
  Extreme Value (GEV) distributions, which also includes the Weibull and
  Frechet.

  The function has a mean of :math:`\mu + 0.57721\beta` and a variance
  of :math:`\frac{\pi^2}{6}\beta^2`.

  References
  ----------
  .. [1] Gumbel, E. J., "Statistics of Extremes,"
         New York: Columbia University Press, 1958.
  .. [2] Reiss, R.-D. and Thomas, M., "Statistical Analysis of Extreme
         Values from Insurance, Finance, Hydrology and Other Fields,"
         Basel: Birkhauser Verlag, 2001.
  """
  return DEFAULT.gumbel(loc, scale, size=size, key=key)


def laplace(loc=None, scale=None, size: Optional[Union[int, Sequence[int]]] = None,
            key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from the Laplace or double exponential distribution with
  specified location (or mean) and scale (decay).

  The Laplace distribution is similar to the Gaussian/normal distribution,
  but is sharper at the peak and has fatter tails. It represents the
  difference between two independent, identically distributed exponential
  random variables.

  Parameters
  ----------
  loc : float or array_like of floats, optional
      The position, :math:`\mu`, of the distribution peak. Default is 0.
  scale : float or array_like of floats, optional
      :math:`\lambda`, the exponential decay. Default is 1. Must be non-
      negative.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``loc`` and ``scale`` are both scalars.
      Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized Laplace distribution.

  Notes
  -----
  It has the probability density function

  .. math:: f(x; \mu, \lambda) = \frac{1}{2\lambda}
                                 \exp\left(-\frac{|x - \mu|}{\lambda}\right).

  The first law of Laplace, from 1774, states that the frequency
  of an error can be expressed as an exponential function of the
  absolute magnitude of the error, which leads to the Laplace
  distribution. For many problems in economics and health
  sciences, this distribution seems to model the data better
  than the standard Gaussian distribution.

  References
  ----------
  .. [1] Abramowitz, M. and Stegun, I. A. (Eds.). "Handbook of
         Mathematical Functions with Formulas, Graphs, and Mathematical
         Tables, 9th printing," New York: Dover, 1972.
  .. [2] Kotz, Samuel, et. al. "The Laplace Distribution and
         Generalizations, " Birkhauser, 2001.
  .. [3] Weisstein, Eric W. "Laplace Distribution."
         From MathWorld--A Wolfram Web Resource.
         http://mathworld.wolfram.com/LaplaceDistribution.html
  .. [4] Wikipedia, "Laplace distribution",
         https://en.wikipedia.org/wiki/Laplace_distribution

  Examples
  --------
  Draw samples from the distribution

  >>> loc, scale = 0., 1.
  >>> s = bm.random.laplace(loc, scale, 1000)

  Display the histogram of the samples, along with
  the probability density function:

  >>> import matplotlib.pyplot as plt
  >>> count, bins, ignored = plt.hist(s, 30, density=True)
  >>> x = np.arange(-8., 8., .01)
  >>> pdf = np.exp(-abs(x-loc)/scale)/(2.*scale)
  >>> plt.plot(x, pdf)

  Plot Gaussian for comparison:

  >>> g = (1/(scale * np.sqrt(2 * np.pi)) *
  ...      np.exp(-(x - loc)**2 / (2 * scale**2)))
  >>> plt.plot(x,g)
  """
  return DEFAULT.laplace(loc, scale, size, key=key)


def logistic(loc=None, scale=None, size: Optional[Union[int, Sequence[int]]] = None,
             key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a logistic distribution.

  Samples are drawn from a logistic distribution with specified
  parameters, loc (location or mean, also median), and scale (>0).

  Parameters
  ----------
  loc : float or array_like of floats, optional
      Parameter of the distribution. Default is 0.
  scale : float or array_like of floats, optional
      Parameter of the distribution. Must be non-negative.
      Default is 1.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``loc`` and ``scale`` are both scalars.
      Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized logistic distribution.

  Notes
  -----
  The probability density for the Logistic distribution is

  .. math:: P(x) = P(x) = \frac{e^{-(x-\mu)/s}}{s(1+e^{-(x-\mu)/s})^2},

  where :math:`\mu` = location and :math:`s` = scale.

  The Logistic distribution is used in Extreme Value problems where it
  can act as a mixture of Gumbel distributions, in Epidemiology, and by
  the World Chess Federation (FIDE) where it is used in the Elo ranking
  system, assuming the performance of each player is a logistically
  distributed random variable.

  References
  ----------
  .. [1] Reiss, R.-D. and Thomas M. (2001), "Statistical Analysis of
         Extreme Values, from Insurance, Finance, Hydrology and Other
         Fields," Birkhauser Verlag, Basel, pp 132-133.
  .. [2] Weisstein, Eric W. "Logistic Distribution." From
         MathWorld--A Wolfram Web Resource.
         http://mathworld.wolfram.com/LogisticDistribution.html
  .. [3] Wikipedia, "Logistic-distribution",
         https://en.wikipedia.org/wiki/Logistic_distribution

  Examples
  --------
  Draw samples from the distribution:

  >>> loc, scale = 10, 1
  >>> s = bm.random.logistic(loc, scale, 10000)
  >>> import matplotlib.pyplot as plt
  >>> count, bins, ignored = plt.hist(s, bins=50)

  #   plot against distribution

  >>> def logist(x, loc, scale):
  ...     return np.exp((loc-x)/scale)/(scale*(1+np.exp((loc-x)/scale))**2)
  >>> lgst_val = logist(bins, loc, scale)
  >>> plt.plot(bins, lgst_val * count.max() / lgst_val.max())
  >>> plt.show()
  """
  return DEFAULT.logistic(loc, scale, size, key=key)


def normal(loc=None, scale=None, size: Optional[Union[int, Sequence[int]]] = None,
           key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw random samples from a normal (Gaussian) distribution.

  The probability density function of the normal distribution, first
  derived by De Moivre and 200 years later by both Gauss and Laplace
  independently [2]_, is often called the bell curve because of
  its characteristic shape (see the example below).

  The normal distributions occurs often in nature.  For example, it
  describes the commonly occurring distribution of samples influenced
  by a large number of tiny, random disturbances, each with its own
  unique distribution [2]_.

  Parameters
  ----------
  loc : float or array_like of floats
      Mean ("centre") of the distribution.
  scale : float or array_like of floats
      Standard deviation (spread or "width") of the distribution. Must be
      non-negative.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``loc`` and ``scale`` are both scalars.
      Otherwise, ``np.broadcast(loc, scale).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized normal distribution.

  Notes
  -----
  The probability density for the Gaussian distribution is

  .. math:: p(x) = \frac{1}{\sqrt{ 2 \pi \sigma^2 }}
                   e^{ - \frac{ (x - \mu)^2 } {2 \sigma^2} },

  where :math:`\mu` is the mean and :math:`\sigma` the standard
  deviation. The square of the standard deviation, :math:`\sigma^2`,
  is called the variance.

  The function has its peak at the mean, and its "spread" increases with
  the standard deviation (the function reaches 0.607 times its maximum at
  :math:`x + \sigma` and :math:`x - \sigma` [2]_).  This implies that
  normal is more likely to return samples lying close to the mean, rather
  than those far away.

  References
  ----------
  .. [1] Wikipedia, "Normal distribution",
         https://en.wikipedia.org/wiki/Normal_distribution
  .. [2] P. R. Peebles Jr., "Central Limit Theorem" in "Probability,
         Random Variables and Random Signal Principles", 4th ed., 2001,
         pp. 51, 51, 125.

  Examples
  --------
  Draw samples from the distribution:

  >>> mu, sigma = 0, 0.1 # mean and standard deviation
  >>> s = bm.random.normal(mu, sigma, 1000)

  Verify the mean and the variance:

  >>> abs(mu - np.mean(s))
  0.0  # may vary

  >>> abs(sigma - np.std(s, ddof=1))
  0.1  # may vary

  Display the histogram of the samples, along with
  the probability density function:

  >>> import matplotlib.pyplot as plt
  >>> count, bins, ignored = plt.hist(s, 30, density=True)
  >>> plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
  ...                np.exp( - (bins - mu)**2 / (2 * sigma**2) ),
  ...          linewidth=2, color='r')
  >>> plt.show()

  Two-by-four array of samples from the normal distribution with
  mean 3 and standard deviation 2.5:

  >>> bm.random.normal(3, 2.5, size=(2, 4))
  array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
         [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random
  """
  return DEFAULT.normal(loc, scale, size, key=key)


def pareto(a, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a Pareto II or Lomax distribution with
  specified shape.

  The Lomax or Pareto II distribution is a shifted Pareto
  distribution. The classical Pareto distribution can be
  obtained from the Lomax distribution by adding 1 and
  multiplying by the scale parameter ``m`` (see Notes).  The
  smallest value of the Lomax distribution is zero while for the
  classical Pareto distribution it is ``mu``, where the standard
  Pareto distribution has location ``mu = 1``.  Lomax can also
  be considered as a simplified version of the Generalized
  Pareto distribution (available in SciPy), with the scale set
  to one and the location set to zero.

  The Pareto distribution must be greater than zero, and is
  unbounded above.  It is also known as the "80-20 rule".  In
  this distribution, 80 percent of the weights are in the lowest
  20 percent of the range, while the other 20 percent fill the
  remaining 80 percent of the range.

  Parameters
  ----------
  a : float or array_like of floats
      Shape of the distribution. Must be positive.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``a`` is a scalar.  Otherwise,
      ``np.array(a).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized Pareto distribution.

  See Also
  --------
  scipy.stats.lomax : probability density function, distribution or
      cumulative density function, etc.
  scipy.stats.genpareto : probability density function, distribution or
      cumulative density function, etc.

  Notes
  -----
  The probability density for the Pareto distribution is

  .. math:: p(x) = \frac{am^a}{x^{a+1}}

  where :math:`a` is the shape and :math:`m` the scale.

  The Pareto distribution, named after the Italian economist
  Vilfredo Pareto, is a power law probability distribution
  useful in many real world problems.  Outside the field of
  economics it is generally referred to as the Bradford
  distribution. Pareto developed the distribution to describe
  the distribution of wealth in an economy.  It has also found
  use in insurance, web page access statistics, oil field sizes,
  and many other problems, including the download frequency for
  projects in Sourceforge [1]_.  It is one of the so-called
  "fat-tailed" distributions.

  References
  ----------
  .. [1] Francis Hunt and Paul Johnson, On the Pareto Distribution of
         Sourceforge projects.
  .. [2] Pareto, V. (1896). Course of Political Economy. Lausanne.
  .. [3] Reiss, R.D., Thomas, M.(2001), Statistical Analysis of Extreme
         Values, Birkhauser Verlag, Basel, pp 23-30.
  .. [4] Wikipedia, "Pareto distribution",
         https://en.wikipedia.org/wiki/Pareto_distribution

  Examples
  --------
  Draw samples from the distribution:

  >>> a, m = 3., 2.  # shape and mode
  >>> s = (bm.random.pareto(a, 1000) + 1) * m

  Display the histogram of the samples, along with the probability
  density function:

  >>> import matplotlib.pyplot as plt
  >>> count, bins, _ = plt.hist(s, 100, density=True)
  >>> fit = a*m**a / bins**(a+1)
  >>> plt.plot(bins, max(count)*fit/max(fit), linewidth=2, color='r')
  >>> plt.show()
  """
  return DEFAULT.pareto(a, size, key=key)


def poisson(lam=1.0, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a Poisson distribution.

  The Poisson distribution is the limit of the binomial distribution
  for large N.

  Parameters
  ----------
  lam : float or array_like of floats
      Expected number of events occurring in a fixed-time interval,
      must be >= 0. A sequence must be broadcastable over the requested
      size.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``lam`` is a scalar. Otherwise,
      ``np.array(lam).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized Poisson distribution.

  Notes
  -----
  The Poisson distribution

  .. math:: f(k; \lambda)=\frac{\lambda^k e^{-\lambda}}{k!}

  For events with an expected separation :math:`\lambda` the Poisson
  distribution :math:`f(k; \lambda)` describes the probability of
  :math:`k` events occurring within the observed
  interval :math:`\lambda`.

  Because the output is limited to the range of the C int64 type, a
  ValueError is raised when `lam` is within 10 sigma of the maximum
  representable value.

  References
  ----------
  .. [1] Weisstein, Eric W. "Poisson Distribution."
         From MathWorld--A Wolfram Web Resource.
         http://mathworld.wolfram.com/PoissonDistribution.html
  .. [2] Wikipedia, "Poisson distribution",
         https://en.wikipedia.org/wiki/Poisson_distribution

  Examples
  --------
  Draw samples from the distribution:

  >>> import numpy as np
  >>> s = bm.random.poisson(5, 10000)

  Display histogram of the sample:

  >>> import matplotlib.pyplot as plt
  >>> count, bins, ignored = plt.hist(s, 14, density=True)
  >>> plt.show()

  Draw each 100 values for lambda 100 and 500:

  >>> s = bm.random.poisson(lam=(100., 500.), size=(100, 2))
  """
  return DEFAULT.poisson(lam, size, key=key)


def standard_cauchy(size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a standard Cauchy distribution with mode = 0.

  Also known as the Lorentz distribution.

  Parameters
  ----------
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  Default is None, in which case a
      single value is returned.

  Returns
  -------
  samples : ndarray or scalar
      The drawn samples.

  Notes
  -----
  The probability density function for the full Cauchy distribution is

  .. math:: P(x; x_0, \gamma) = \frac{1}{\pi \gamma \bigl[ 1+
            (\frac{x-x_0}{\gamma})^2 \bigr] }

  and the Standard Cauchy distribution just sets :math:`x_0=0` and
  :math:`\gamma=1`

  The Cauchy distribution arises in the solution to the driven harmonic
  oscillator problem, and also describes spectral line broadening. It
  also describes the distribution of values at which a line tilted at
  a random angle will cut the x axis.

  When studying hypothesis tests that assume normality, seeing how the
  tests perform on data from a Cauchy distribution is a good indicator of
  their sensitivity to a heavy-tailed distribution, since the Cauchy looks
  very much like a Gaussian distribution, but with heavier tails.

  References
  ----------
  .. [1] NIST/SEMATECH e-Handbook of Statistical Methods, "Cauchy
        Distribution",
        https://www.itl.nist.gov/div898/handbook/eda/section3/eda3663.htm
  .. [2] Weisstein, Eric W. "Cauchy Distribution." From MathWorld--A
        Wolfram Web Resource.
        http://mathworld.wolfram.com/CauchyDistribution.html
  .. [3] Wikipedia, "Cauchy distribution"
        https://en.wikipedia.org/wiki/Cauchy_distribution

  Examples
  --------
  Draw samples and plot the distribution:

  >>> import matplotlib.pyplot as plt
  >>> s = bm.random.standard_cauchy(1000000)
  >>> s = s[(s>-25) & (s<25)]  # truncate distribution so it plots well
  >>> plt.hist(s, bins=100)
  >>> plt.show()
  """
  return DEFAULT.standard_cauchy(size, key=key)


def standard_exponential(size: Optional[Union[int, Sequence[int]]] = None,
                         key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from the standard exponential distribution.

  `standard_exponential` is identical to the exponential distribution
  with a scale parameter of 1.

  Parameters
  ----------
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  Default is None, in which case a
      single value is returned.

  Returns
  -------
  out : float or ndarray
      Drawn samples.

  Examples
  --------
  Output a 3x8000 array:

  >>> n = bm.random.standard_exponential((3, 8000))
  """
  return DEFAULT.standard_exponential(size, key=key)


def standard_gamma(shape, size: Optional[Union[int, Sequence[int]]] = None,
                   key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a standard Gamma distribution.

  Samples are drawn from a Gamma distribution with specified parameters,
  shape (sometimes designated "k") and scale=1.

  Parameters
  ----------
  shape : float or array_like of floats
      Parameter, must be non-negative.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``shape`` is a scalar.  Otherwise,
      ``np.array(shape).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized standard gamma distribution.

  See Also
  --------
  scipy.stats.gamma : probability density function, distribution or
      cumulative density function, etc.

  Notes
  -----
  The probability density for the Gamma distribution is

  .. math:: p(x) = x^{k-1}\frac{e^{-x/\theta}}{\theta^k\Gamma(k)},

  where :math:`k` is the shape and :math:`\theta` the scale,
  and :math:`\Gamma` is the Gamma function.

  The Gamma distribution is often used to model the times to failure of
  electronic components, and arises naturally in processes for which the
  waiting times between Poisson distributed events are relevant.

  References
  ----------
  .. [1] Weisstein, Eric W. "Gamma Distribution." From MathWorld--A
         Wolfram Web Resource.
         http://mathworld.wolfram.com/GammaDistribution.html
  .. [2] Wikipedia, "Gamma distribution",
         https://en.wikipedia.org/wiki/Gamma_distribution

  Examples
  --------
  Draw samples from the distribution:

  >>> shape, scale = 2., 1. # mean and width
  >>> s = bm.random.standard_gamma(shape, 1000000)

  Display the histogram of the samples, along with
  the probability density function:

  >>> import matplotlib.pyplot as plt
  >>> import scipy.special as sps  # doctest: +SKIP
  >>> count, bins, ignored = plt.hist(s, 50, density=True)
  >>> y = bins**(shape-1) * ((np.exp(-bins/scale))/  # doctest: +SKIP
  ...                       (sps.gamma(shape) * scale**shape))
  >>> plt.plot(bins, y, linewidth=2, color='r')  # doctest: +SKIP
  >>> plt.show()
  """
  return DEFAULT.standard_gamma(shape, size, key=key)


def standard_normal(size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a standard Normal distribution (mean=0, stdev=1).

  Parameters
  ----------
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  Default is None, in which case a
      single value is returned.

  Returns
  -------
  out : float or ndarray
      A floating-point array of shape ``size`` of drawn samples, or a
      single sample if ``size`` was not specified.

  See Also
  --------
  normal :
      Equivalent function with additional ``loc`` and ``scale`` arguments
      for setting the mean and standard deviation.

  Notes
  -----
  For random samples from the normal distribution with mean ``mu`` and
  standard deviation ``sigma``, use one of::

      mu + sigma * bm.random.standard_normal(size=...)
      bm.random.normal(mu, sigma, size=...)

  Examples
  --------
  >>> bm.random.standard_normal()
  2.1923875335537315 #random

  >>> s = bm.random.standard_normal(8000)
  >>> s
  array([ 0.6888893 ,  0.78096262, -0.89086505, ...,  0.49876311,  # random
         -0.38672696, -0.4685006 ])                                # random
  >>> s.shape
  (8000,)
  >>> s = bm.random.standard_normal(size=(3, 4, 2))
  >>> s.shape
  (3, 4, 2)

  Two-by-four array of samples from the normal distribution with
  mean 3 and standard deviation 2.5:

  >>> 3 + 2.5 * bm.random.standard_normal(size=(2, 4))
  array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
         [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random
  """
  return DEFAULT.standard_normal(size, key=key)


def standard_t(df, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a standard Student's t distribution with `df` degrees
  of freedom.

  A special case of the hyperbolic distribution.  As `df` gets
  large, the result resembles that of the standard normal
  distribution (`standard_normal`).

  Parameters
  ----------
  df : float or array_like of floats
      Degrees of freedom, must be > 0.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``df`` is a scalar.  Otherwise,
      ``np.array(df).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized standard Student's t distribution.

  Notes
  -----
  The probability density function for the t distribution is

  .. math:: P(x, df) = \frac{\Gamma(\frac{df+1}{2})}{\sqrt{\pi df}
            \Gamma(\frac{df}{2})}\Bigl( 1+\frac{x^2}{df} \Bigr)^{-(df+1)/2}

  The t test is based on an assumption that the data come from a
  Normal distribution. The t test provides a way to test whether
  the sample mean (that is the mean calculated from the data) is
  a good estimate of the true mean.

  The derivation of the t-distribution was first published in
  1908 by William Gosset while working for the Guinness Brewery
  in Dublin. Due to proprietary issues, he had to publish under
  a pseudonym, and so he used the name Student.

  References
  ----------
  .. [1] Dalgaard, Peter, "Introductory Statistics With R",
         Springer, 2002.
  .. [2] Wikipedia, "Student's t-distribution"
         https://en.wikipedia.org/wiki/Student's_t-distribution

  Examples
  --------
  From Dalgaard page 83 [1]_, suppose the daily energy intake for 11
  women in kilojoules (kJ) is:

  >>> intake = np.array([5260., 5470, 5640, 6180, 6390, 6515, 6805, 7515, \
  ...                    7515, 8230, 8770])

  Does their energy intake deviate systematically from the recommended
  value of 7725 kJ? Our null hypothesis will be the absence of deviation,
  and the alternate hypothesis will be the presence of an effect that could be
  either positive or negative, hence making our test 2-tailed.

  Because we are estimating the mean and we have N=11 values in our sample,
  we have N-1=10 degrees of freedom. We set our significance level to 95% and
  compute the t statistic using the empirical mean and empirical standard
  deviation of our intake. We use a ddof of 1 to base the computation of our
  empirical standard deviation on an unbiased estimate of the variance (note:
  the final estimate is not unbiased due to the concave nature of the square
  root).

  >>> np.mean(intake)
  6753.636363636364
  >>> intake.std(ddof=1)
  1142.1232221373727
  >>> t = (np.mean(intake)-7725)/(intake.std(ddof=1)/np.sqrt(len(intake)))
  >>> t
  -2.8207540608310198

  We draw 1000000 samples from Student's t distribution with the adequate
  degrees of freedom.

  >>> import matplotlib.pyplot as plt
  >>> s = bm.random.standard_t(10, size=1000000)
  >>> h = plt.hist(s, bins=100, density=True)

  Does our t statistic land in one of the two critical regions found at
  both tails of the distribution?

  >>> np.sum(np.abs(t) < np.abs(s)) / float(len(s))
  0.018318  #random < 0.05, statistic is in critical region

  The probability value for this 2-tailed test is about 1.83%, which is
  lower than the 5% pre-determined significance threshold.

  Therefore, the probability of observing values as extreme as our intake
  conditionally on the null hypothesis being true is too low, and we reject
  the null hypothesis of no deviation.
  """
  return DEFAULT.standard_t(df, size, key=key)


def uniform(low=0.0, high=1.0, size: Optional[Union[int, Sequence[int]]] = None,
            key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a uniform distribution.

  Samples are uniformly distributed over the half-open interval
  ``[low, high)`` (includes low, but excludes high).  In other words,
  any value within the given interval is equally likely to be drawn
  by `uniform`.

  Parameters
  ----------
  low : float or array_like of floats, optional
      Lower boundary of the output interval.  All values generated will be
      greater than or equal to low.  The default value is 0.
  high : float or array_like of floats
      Upper boundary of the output interval.  All values generated will be
      less than or equal to high.  The high limit may be included in the
      returned array of floats due to floating-point rounding in the
      equation ``low + (high-low) * random_sample()``.  The default value
      is 1.0.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``low`` and ``high`` are both scalars.
      Otherwise, ``np.broadcast(low, high).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized uniform distribution.

  See Also
  --------
  randint : Discrete uniform distribution, yielding integers.
  random_integers : Discrete uniform distribution over the closed
                    interval ``[low, high]``.
  random_sample : Floats uniformly distributed over ``[0, 1)``.
  random : Alias for `random_sample`.
  rand : Convenience function that accepts dimensions as input, e.g.,
         ``rand(2,2)`` would generate a 2-by-2 array of floats,
         uniformly distributed over ``[0, 1)``.

  Notes
  -----
  The probability density function of the uniform distribution is

  .. math:: p(x) = \frac{1}{b - a}

  anywhere within the interval ``[a, b)``, and zero elsewhere.

  When ``high`` == ``low``, values of ``low`` will be returned.
  If ``high`` < ``low``, the results are officially undefined
  and may eventually raise an error, i.e. do not rely on this
  function to behave when passed arguments satisfying that
  inequality condition. The ``high`` limit may be included in the
  returned array of floats due to floating-point rounding in the
  equation ``low + (high-low) * random_sample()``. For example:

  >>> x = np.float32(5*0.99999999)
  >>> x
  5.0


  Examples
  --------
  Draw samples from the distribution:

  >>> s = bm.random.uniform(-1,0,1000)

  All values are within the given interval:

  >>> np.all(s >= -1)
  True
  >>> np.all(s < 0)
  True

  Display the histogram of the samples, along with the
  probability density function:

  >>> import matplotlib.pyplot as plt
  >>> count, bins, ignored = plt.hist(s, 15, density=True)
  >>> plt.plot(bins, np.ones_like(bins), linewidth=2, color='r')
  >>> plt.show()
  """
  return DEFAULT.uniform(low, high, size, key=key)


def truncated_normal(lower, upper, size: Optional[Union[int, Sequence[int]]] = None, loc=0., scale=1., dtype=float,
                     key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""Sample truncated standard normal random values with given shape and dtype.

  Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf


  Notes
  -----
  This distribution is the normal distribution centered on ``loc`` (default
  0), with standard deviation ``scale`` (default 1), and clipped at ``a``,
  ``b`` standard deviations to the left, right (respectively) from ``loc``.
  If ``myclip_a`` and ``myclip_b`` are clip values in the sample space (as
  opposed to the number of standard deviations) then they can be converted
  to the required form according to::

      a, b = (myclip_a - loc) / scale, (myclip_b - loc) / scale


  Parameters
  ----------
  lower : float, ndarray
    A float or array of floats representing the lower bound for
    truncation. Must be broadcast-compatible with ``upper``.
  upper : float, ndarray
    A float or array of floats representing the  upper bound for
    truncation. Must be broadcast-compatible with ``lower``.
  loc : float, ndarray
    Mean ("centre") of the distribution before truncating. Note that 
    the mean of the truncated distribution will not be exactly equal 
    to ``loc``.
  size : optional, list of int, tuple of int
    A tuple of nonnegative integers specifying the result
    shape. Must be broadcast-compatible with ``lower`` and ``upper``. The
    default (None) produces a result shape by broadcasting ``lower`` and
    ``upper``.
  loc: optional, float, ndarray
    A float or array of floats representing the mean of the
    distribution. Default is 0.
  scale : float, ndarray
    Standard deviation (spread or "width") of the distribution. Must be
    non-negative. Default is 1.
  dtype: optional
    The float dtype for the returned values (default float64 if
    jax_enable_x64 is true, otherwise float32).
  key: jax.Array
    The key for random generator. Consistent with the jax's random
    paradigm.

  Returns
  -------
  out : Array
    A random array with the specified dtype and shape given by ``shape`` if
    ``shape`` is not None, or else by broadcasting ``lower`` and ``upper``.
    Returns values in the open interval ``(lower, upper)``.
  """
  return DEFAULT.truncated_normal(lower, upper, size, loc, scale, dtype=dtype, key=key)


RandomState.truncated_normal.__doc__ = truncated_normal.__doc__


def bernoulli(p=0.5, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""Sample Bernoulli random values with given shape and mean.

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


def lognormal(mean=None, sigma=None, size: Optional[Union[int, Sequence[int]]] = None,
              key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a log-normal distribution.

  Draw samples from a log-normal distribution with specified mean,
  standard deviation, and array shape.  Note that the mean and standard
  deviation are not the values for the distribution itself, but of the
  underlying normal distribution it is derived from.

  Parameters
  ----------
  mean : float or array_like of floats, optional
      Mean value of the underlying normal distribution. Default is 0.
  sigma : float or array_like of floats, optional
      Standard deviation of the underlying normal distribution. Must be
      non-negative. Default is 1.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``mean`` and ``sigma`` are both scalars.
      Otherwise, ``np.broadcast(mean, sigma).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized log-normal distribution.

  See Also
  --------
  scipy.stats.lognorm : probability density function, distribution,
      cumulative density function, etc.

  Notes
  -----
  A variable `x` has a log-normal distribution if `log(x)` is normally
  distributed.  The probability density function for the log-normal
  distribution is:

  .. math:: p(x) = \frac{1}{\sigma x \sqrt{2\pi}}
                   e^{(-\frac{(ln(x)-\mu)^2}{2\sigma^2})}

  where :math:`\mu` is the mean and :math:`\sigma` is the standard
  deviation of the normally distributed logarithm of the variable.
  A log-normal distribution results if a random variable is the *product*
  of a large number of independent, identically-distributed variables in
  the same way that a normal distribution results if the variable is the
  *sum* of a large number of independent, identically-distributed
  variables.

  References
  ----------
  .. [1] Limpert, E., Stahel, W. A., and Abbt, M., "Log-normal
         Distributions across the Sciences: Keys and Clues,"
         BioScience, Vol. 51, No. 5, May, 2001.
         https://stat.ethz.ch/~stahel/lognormal/bioscience.pdf
  .. [2] Reiss, R.D. and Thomas, M., "Statistical Analysis of Extreme
         Values," Basel: Birkhauser Verlag, 2001, pp. 31-32.

  Examples
  --------
  Draw samples from the distribution:

  >>> mu, sigma = 3., 1. # mean and standard deviation
  >>> s = bm.random.lognormal(mu, sigma, 1000)

  Display the histogram of the samples, along with
  the probability density function:

  >>> import matplotlib.pyplot as plt
  >>> count, bins, ignored = plt.hist(s, 100, density=True, align='mid')

  >>> x = np.linspace(min(bins), max(bins), 10000)
  >>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
  ...        / (x * sigma * np.sqrt(2 * np.pi)))

  >>> plt.plot(x, pdf, linewidth=2, color='r')
  >>> plt.axis('tight')
  >>> plt.show()

  Demonstrate that taking the products of random samples from a uniform
  distribution can be fit well by a log-normal probability density
  function.

  >>> # Generate a thousand samples: each is the product of 100 random
  >>> # values, drawn from a normal distribution.
  >>> b = []
  >>> for i in range(1000):
  ...    a = 10. + bm.random.standard_normal(100)
  ...    b.append(np.product(a))

  >>> b = np.array(b) / np.min(b) # scale values to be positive
  >>> count, bins, ignored = plt.hist(b, 100, density=True, align='mid')
  >>> sigma = np.std(np.log(b))
  >>> mu = np.mean(np.log(b))

  >>> x = np.linspace(min(bins), max(bins), 10000)
  >>> pdf = (np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2))
  ...        / (x * sigma * np.sqrt(2 * np.pi)))

  >>> plt.plot(x, pdf, color='r', linewidth=2)
  >>> plt.show()
  """
  return DEFAULT.lognormal(mean, sigma, size, key=key)


def binomial(n, p, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a binomial distribution.

  Samples are drawn from a binomial distribution with specified
  parameters, n trials and p probability of success where
  n an integer >= 0 and p is in the interval [0,1]. (n may be
  input as a float, but it is truncated to an integer in use)

  Parameters
  ----------
  n : int or array_like of ints
      Parameter of the distribution, >= 0. Floats are also accepted,
      but they will be truncated to integers.
  p : float or array_like of floats
      Parameter of the distribution, >= 0 and <=1.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``n`` and ``p`` are both scalars.
      Otherwise, ``np.broadcast(n, p).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized binomial distribution, where
      each sample is equal to the number of successes over the n trials.

  See Also
  --------
  scipy.stats.binom : probability density function, distribution or
      cumulative density function, etc.

  Notes
  -----
  The probability density for the binomial distribution is

  .. math:: P(N) = \binom{n}{N}p^N(1-p)^{n-N},

  where :math:`n` is the number of trials, :math:`p` is the probability
  of success, and :math:`N` is the number of successes.

  When estimating the standard error of a proportion in a population by
  using a random sample, the normal distribution works well unless the
  product p*n <=5, where p = population proportion estimate, and n =
  number of samples, in which case the binomial distribution is used
  instead. For example, a sample of 15 people shows 4 who are left
  handed, and 11 who are right handed. Then p = 4/15 = 27%. 0.27*15 = 4,
  so the binomial distribution should be used in this case.

  References
  ----------
  .. [1] Dalgaard, Peter, "Introductory Statistics with R",
         Springer-Verlag, 2002.
  .. [2] Glantz, Stanton A. "Primer of Biostatistics.", McGraw-Hill,
         Fifth Edition, 2002.
  .. [3] Lentner, Marvin, "Elementary Applied Statistics", Bogden
         and Quigley, 1972.
  .. [4] Weisstein, Eric W. "Binomial Distribution." From MathWorld--A
         Wolfram Web Resource.
         http://mathworld.wolfram.com/BinomialDistribution.html
  .. [5] Wikipedia, "Binomial distribution",
         https://en.wikipedia.org/wiki/Binomial_distribution

  Examples
  --------
  Draw samples from the distribution:

  >>> n, p = 10, .5  # number of trials, probability of each trial
  >>> s = bm.random.binomial(n, p, 1000)
  # result of flipping a coin 10 times, tested 1000 times.

  A real world example. A company drills 9 wild-cat oil exploration
  wells, each with an estimated probability of success of 0.1. All nine
  wells fail. What is the probability of that happening?

  Let's do 20,000 trials of the model, and count the number that
  generate zero positive results.

  >>> sum(bm.random.binomial(9, 0.1, 20000) == 0)/20000.
  # answer = 0.38885, or 38%.
  """
  return DEFAULT.binomial(n, p, size, key=key)


def chisquare(df, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a chi-square distribution.

  When `df` independent random variables, each with standard normal
  distributions (mean 0, variance 1), are squared and summed, the
  resulting distribution is chi-square (see Notes).  This distribution
  is often used in hypothesis testing.

  Parameters
  ----------
  df : float or array_like of floats
       Number of degrees of freedom, must be > 0.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``df`` is a scalar.  Otherwise,
      ``np.array(df).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized chi-square distribution.

  Raises
  ------
  ValueError
      When `df` <= 0 or when an inappropriate `size` (e.g. ``size=-1``)
      is given.

  Notes
  -----
  The variable obtained by summing the squares of `df` independent,
  standard normally distributed random variables:

  .. math:: Q = \sum_{i=0}^{\mathtt{df}} X^2_i

  is chi-square distributed, denoted

  .. math:: Q \sim \chi^2_k.

  The probability density function of the chi-squared distribution is

  .. math:: p(x) = \frac{(1/2)^{k/2}}{\Gamma(k/2)}
                   x^{k/2 - 1} e^{-x/2},

  where :math:`\Gamma` is the gamma function,

  .. math:: \Gamma(x) = \int_0^{-\infty} t^{x - 1} e^{-t} dt.

  References
  ----------
  .. [1] NIST "Engineering Statistics Handbook"
         https://www.itl.nist.gov/div898/handbook/eda/section3/eda3666.htm

  Examples
  --------
  >>> bm.random.chisquare(2,4)
  array([ 1.89920014,  9.00867716,  3.13710533,  5.62318272]) # random
  """
  return DEFAULT.chisquare(df, size, key=key)


def dirichlet(alpha, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from the Dirichlet distribution.

  Draw `size` samples of dimension k from a Dirichlet distribution. A
  Dirichlet-distributed random variable can be seen as a multivariate
  generalization of a Beta distribution. The Dirichlet distribution
  is a conjugate prior of a multinomial distribution in Bayesian
  inference.

  Parameters
  ----------
  alpha : sequence of floats, length k
      Parameter of the distribution (length ``k`` for sample of
      length ``k``).
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n)``, then
      ``m * n * k`` samples are drawn.  Default is None, in which case a
      vector of length ``k`` is returned.

  Returns
  -------
  samples : ndarray,
      The drawn samples, of shape ``(size, k)``.

  Raises
  ------
  ValueError
      If any value in ``alpha`` is less than or equal to zero

  Notes
  -----
  The Dirichlet distribution is a distribution over vectors
  :math:`x` that fulfil the conditions :math:`x_i>0` and
  :math:`\sum_{i=1}^k x_i = 1`.

  The probability density function :math:`p` of a
  Dirichlet-distributed random vector :math:`X` is
  proportional to

  .. math:: p(x) \propto \prod_{i=1}^{k}{x^{\alpha_i-1}_i},

  where :math:`\alpha` is a vector containing the positive
  concentration parameters.

  The method uses the following property for computation: let :math:`Y`
  be a random vector which has components that follow a standard gamma
  distribution, then :math:`X = \frac{1}{\sum_{i=1}^k{Y_i}} Y`
  is Dirichlet-distributed

  References
  ----------
  .. [1] David McKay, "Information Theory, Inference and Learning
         Algorithms," chapter 23,
         http://www.inference.org.uk/mackay/itila/
  .. [2] Wikipedia, "Dirichlet distribution",
         https://en.wikipedia.org/wiki/Dirichlet_distribution

  Examples
  --------
  Taking an example cited in Wikipedia, this distribution can be used if
  one wanted to cut strings (each of initial length 1.0) into K pieces
  with different lengths, where each piece had, on average, a designated
  average length, but allowing some variation in the relative sizes of
  the pieces.

  >>> s = bm.random.dirichlet((10, 5, 3), 20).transpose()

  >>> import matplotlib.pyplot as plt
  >>> plt.barh(range(20), s[0])
  >>> plt.barh(range(20), s[1], left=s[0], color='g')
  >>> plt.barh(range(20), s[2], left=s[0]+s[1], color='r')
  >>> plt.title("Lengths of Strings")
  """
  return DEFAULT.dirichlet(alpha, size, key=key)


def geometric(p, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from the geometric distribution.

  Bernoulli trials are experiments with one of two outcomes:
  success or failure (an example of such an experiment is flipping
  a coin).  The geometric distribution models the number of trials
  that must be run in order to achieve success.  It is therefore
  supported on the positive integers, ``k = 1, 2, ...``.

  The probability mass function of the geometric distribution is

  .. math:: f(k) = (1 - p)^{k - 1} p

  where `p` is the probability of success of an individual trial.

  Parameters
  ----------
  p : float or array_like of floats
      The probability of success of an individual trial.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``p`` is a scalar.  Otherwise,
      ``np.array(p).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized geometric distribution.

  Examples
  --------
  Draw ten thousand values from the geometric distribution,
  with the probability of an individual success equal to 0.35:

  >>> z = bm.random.geometric(p=0.35, size=10000)

  How many trials succeeded after a single run?

  >>> (z == 1).sum() / 10000.
  0.34889999999999999 #random
  """
  return DEFAULT.geometric(p, size, key=key)


def f(dfnum, dfden, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from an F distribution.

  Samples are drawn from an F distribution with specified parameters,
  `dfnum` (degrees of freedom in numerator) and `dfden` (degrees of
  freedom in denominator), where both parameters must be greater than
  zero.

  The random variate of the F distribution (also known as the
  Fisher distribution) is a continuous probability distribution
  that arises in ANOVA tests, and is the ratio of two chi-square
  variates.

  Parameters
  ----------
  dfnum : float or array_like of floats
      Degrees of freedom in numerator, must be > 0.
  dfden : float or array_like of float
      Degrees of freedom in denominator, must be > 0.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``dfnum`` and ``dfden`` are both scalars.
      Otherwise, ``np.broadcast(dfnum, dfden).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized Fisher distribution.

  See Also
  --------
  scipy.stats.f : probability density function, distribution or
      cumulative density function, etc.

  Notes
  -----
  The F statistic is used to compare in-group variances to between-group
  variances. Calculating the distribution depends on the sampling, and
  so it is a function of the respective degrees of freedom in the
  problem.  The variable `dfnum` is the number of samples minus one, the
  between-groups degrees of freedom, while `dfden` is the within-groups
  degrees of freedom, the sum of the number of samples in each group
  minus the number of groups.

  References
  ----------
  .. [1] Glantz, Stanton A. "Primer of Biostatistics.", McGraw-Hill,
         Fifth Edition, 2002.
  .. [2] Wikipedia, "F-distribution",
         https://en.wikipedia.org/wiki/F-distribution

  Examples
  --------
  An example from Glantz[1], pp 47-40:

  Two groups, children of diabetics (25 people) and children from people
  without diabetes (25 controls). Fasting blood glucose was measured,
  case group had a mean value of 86.1, controls had a mean value of
  82.2. Standard deviations were 2.09 and 2.49 respectively. Are these
  data consistent with the null hypothesis that the parents diabetic
  status does not affect their children's blood glucose levels?
  Calculating the F statistic from the data gives a value of 36.01.

  Draw samples from the distribution:

  >>> dfnum = 1. # between group degrees of freedom
  >>> dfden = 48. # within groups degrees of freedom
  >>> s = bm.random.f(dfnum, dfden, 1000)

  The lower bound for the top 1% of the samples is :

  >>> np.sort(s)[-10]
  7.61988120985 # random

  So there is about a 1% chance that the F statistic will exceed 7.62,
  the measured value is 36, so the null hypothesis is rejected at the 1%
  level.
  """
  return DEFAULT.f(dfnum, dfden, size, key=key)


def hypergeometric(ngood, nbad, nsample, size: Optional[Union[int, Sequence[int]]] = None,
                   key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a Hypergeometric distribution.

  Samples are drawn from a hypergeometric distribution with specified
  parameters, `ngood` (ways to make a good selection), `nbad` (ways to make
  a bad selection), and `nsample` (number of items sampled, which is less
  than or equal to the sum ``ngood + nbad``).

  Parameters
  ----------
  ngood : int or array_like of ints
      Number of ways to make a good selection.  Must be nonnegative.
  nbad : int or array_like of ints
      Number of ways to make a bad selection.  Must be nonnegative.
  nsample : int or array_like of ints
      Number of items sampled.  Must be at least 1 and at most
      ``ngood + nbad``.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if `ngood`, `nbad`, and `nsample`
      are all scalars.  Otherwise, ``np.broadcast(ngood, nbad, nsample).size``
      samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized hypergeometric distribution. Each
      sample is the number of good items within a randomly selected subset of
      size `nsample` taken from a set of `ngood` good items and `nbad` bad items.

  See Also
  --------
  scipy.stats.hypergeom : probability density function, distribution or
      cumulative density function, etc.

  Notes
  -----
  The probability density for the Hypergeometric distribution is

  .. math:: P(x) = \frac{\binom{g}{x}\binom{b}{n-x}}{\binom{g+b}{n}},

  where :math:`0 \le x \le n` and :math:`n-b \le x \le g`

  for P(x) the probability of ``x`` good results in the drawn sample,
  g = `ngood`, b = `nbad`, and n = `nsample`.

  Consider an urn with black and white marbles in it, `ngood` of them
  are black and `nbad` are white. If you draw `nsample` balls without
  replacement, then the hypergeometric distribution describes the
  distribution of black balls in the drawn sample.

  Note that this distribution is very similar to the binomial
  distribution, except that in this case, samples are drawn without
  replacement, whereas in the Binomial case samples are drawn with
  replacement (or the sample space is infinite). As the sample space
  becomes large, this distribution approaches the binomial.

  References
  ----------
  .. [1] Lentner, Marvin, "Elementary Applied Statistics", Bogden
         and Quigley, 1972.
  .. [2] Weisstein, Eric W. "Hypergeometric Distribution." From
         MathWorld--A Wolfram Web Resource.
         http://mathworld.wolfram.com/HypergeometricDistribution.html
  .. [3] Wikipedia, "Hypergeometric distribution",
         https://en.wikipedia.org/wiki/Hypergeometric_distribution

  Examples
  --------
  Draw samples from the distribution:

  >>> ngood, nbad, nsamp = 100, 2, 10
  # number of good, number of bad, and number of samples
  >>> s = bm.random.hypergeometric(ngood, nbad, nsamp, 1000)
  >>> from matplotlib.pyplot import hist
  >>> hist(s)
  #   note that it is very unlikely to grab both bad items

  Suppose you have an urn with 15 white and 15 black marbles.
  If you pull 15 marbles at random, how likely is it that
  12 or more of them are one color?

  >>> s = bm.random.hypergeometric(15, 15, 15, 100000)
  >>> sum(s>=12)/100000. + sum(s<=3)/100000.
  #   answer = 0.003 ... pretty unlikely!
  """
  return DEFAULT.hypergeometric(ngood, nbad, nsample, size, key=key)


def logseries(p, size: Optional[Union[int, Sequence[int]]] = None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a logarithmic series distribution.

  Samples are drawn from a log series distribution with specified
  shape parameter, 0 <= ``p`` < 1.

  Parameters
  ----------
  p : float or array_like of floats
      Shape parameter for the distribution.  Must be in the range [0, 1).
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``p`` is a scalar.  Otherwise,
      ``np.array(p).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized logarithmic series distribution.

  See Also
  --------
  scipy.stats.logser : probability density function, distribution or
      cumulative density function, etc.

  Notes
  -----
  The probability density for the Log Series distribution is

  .. math:: P(k) = \frac{-p^k}{k \ln(1-p)},

  where p = probability.

  The log series distribution is frequently used to represent species
  richness and occurrence, first proposed by Fisher, Corbet, and
  Williams in 1943 [2].  It may also be used to model the numbers of
  occupants seen in cars [3].

  References
  ----------
  .. [1] Buzas, Martin A.; Culver, Stephen J.,  Understanding regional
         species diversity through the log series distribution of
         occurrences: BIODIVERSITY RESEARCH Diversity & Distributions,
         Volume 5, Number 5, September 1999 , pp. 187-195(9).
  .. [2] Fisher, R.A,, A.S. Corbet, and C.B. Williams. 1943. The
         relation between the number of species and the number of
         individuals in a random sample of an animal population.
         Journal of Animal Ecology, 12:42-58.
  .. [3] D. J. Hand, F. Daly, D. Lunn, E. Ostrowski, A Handbook of Small
         Data Sets, CRC Press, 1994.
  .. [4] Wikipedia, "Logarithmic distribution",
         https://en.wikipedia.org/wiki/Logarithmic_distribution

  Examples
  --------
  Draw samples from the distribution:

  >>> a = .6
  >>> s = bm.random.logseries(a, 10000)
  >>> import matplotlib.pyplot as plt
  >>> count, bins, ignored = plt.hist(s)

  #   plot against distribution

  >>> def logseries(k, p):
  ...     return -p**k/(k*np.log(1-p))
  >>> plt.plot(bins, logseries(bins, a)*count.max()/
  ...          logseries(bins, a).max(), 'r')
  >>> plt.show()
  """
  return DEFAULT.logseries(p, size, key=key)


def multinomial(n, pvals, size: Optional[Union[int, Sequence[int]]] = None,
                key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a multinomial distribution.

  The multinomial distribution is a multivariate generalization of the
  binomial distribution.  Take an experiment with one of ``p``
  possible outcomes.  An example of such an experiment is throwing a dice,
  where the outcome can be 1 through 6.  Each sample drawn from the
  distribution represents `n` such experiments.  Its values,
  ``X_i = [X_0, X_1, ..., X_p]``, represent the number of times the
  outcome was ``i``.

  Parameters
  ----------
  n : int
      Number of experiments.
  pvals : sequence of floats, length p
      Probabilities of each of the ``p`` different outcomes.  These
      must sum to 1 (however, the last element is always assumed to
      account for the remaining probability, as long as
      ``sum(pvals[:-1]) <= 1)``.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  Default is None, in which case a
      single value is returned.

  Returns
  -------
  out : ndarray
      The drawn samples, of shape *size*, if that was provided.  If not,
      the shape is ``(N,)``.

      In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
      value drawn from the distribution.

  Examples
  --------
  Throw a dice 20 times:

  >>> bm.random.multinomial(20, [1/6.]*6, size=1)
  array([[4, 1, 7, 5, 2, 1]]) # random

  It landed 4 times on 1, once on 2, etc.

  Now, throw the dice 20 times, and 20 times again:

  >>> bm.random.multinomial(20, [1/6.]*6, size=2)
  array([[3, 4, 3, 3, 4, 3], # random
         [2, 4, 3, 4, 0, 7]])

  For the first run, we threw 3 times 1, 4 times 2, etc.  For the second,
  we threw 2 times 1, 4 times 2, etc.

  A loaded die is more likely to land on number 6:

  >>> bm.random.multinomial(100, [1/7.]*5 + [2/7.])
  array([11, 16, 14, 17, 16, 26]) # random

  The probability inputs should be normalized. As an implementation
  detail, the value of the last entry is ignored and assumed to take
  up any leftover probability mass, but this should not be relied on.
  A biased coin which has twice as much weight on one side as on the
  other should be sampled like so:

  >>> bm.random.multinomial(100, [1.0 / 3, 2.0 / 3])  # RIGHT
  array([38, 62]) # random

  not like:

  >>> bm.random.multinomial(100, [1.0, 2.0])  # WRONG
  Traceback (most recent call last):
  ValueError: pvals < 0, pvals > 1 or pvals contains NaNs
  """
  return DEFAULT.multinomial(n, pvals, size, key=key)


def multivariate_normal(mean, cov, size: Optional[Union[int, Sequence[int]]] = None, method: str = 'cholesky',
                        key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw random samples from a multivariate normal distribution.

  The multivariate normal, multinormal or Gaussian distribution is a
  generalization of the one-dimensional normal distribution to higher
  dimensions.  Such a distribution is specified by its mean and
  covariance matrix.  These parameters are analogous to the mean
  (average or "center") and variance (standard deviation, or "width,"
  squared) of the one-dimensional normal distribution.

  Parameters
  ----------
  mean : 1-D array_like, of length N
      Mean of the N-dimensional distribution.
  cov : 2-D array_like, of shape (N, N)
      Covariance matrix of the distribution. It must be symmetric and
      positive-semidefinite for proper sampling.
  size : int or tuple of ints, optional
      Given a shape of, for example, ``(m,n,k)``, ``m*n*k`` samples are
      generated, and packed in an `m`-by-`n`-by-`k` arrangement.  Because
      each sample is `N`-dimensional, the output shape is ``(m,n,k,N)``.
      If no shape is specified, a single (`N`-D) sample is returned.
  check_valid : { 'warn', 'raise', 'ignore' }, optional
      Behavior when the covariance matrix is not positive semidefinite.
  tol : float, optional
      Tolerance when checking the singular values in covariance matrix.
      cov is cast to double before the check.

  Returns
  -------
  out : ndarray
      The drawn samples, of shape *size*, if that was provided.  If not,
      the shape is ``(N,)``.

      In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
      value drawn from the distribution.

  Notes
  -----
  The mean is a coordinate in N-dimensional space, which represents the
  location where samples are most likely to be generated.  This is
  analogous to the peak of the bell curve for the one-dimensional or
  univariate normal distribution.

  Covariance indicates the level to which two variables vary together.
  From the multivariate normal distribution, we draw N-dimensional
  samples, :math:`X = [x_1, x_2, ... x_N]`.  The covariance matrix
  element :math:`C_{ij}` is the covariance of :math:`x_i` and :math:`x_j`.
  The element :math:`C_{ii}` is the variance of :math:`x_i` (i.e. its
  "spread").

  Instead of specifying the full covariance matrix, popular
  approximations include:

    - Spherical covariance (`cov` is a multiple of the identity matrix)
    - Diagonal covariance (`cov` has non-negative elements, and only on
      the diagonal)

  This geometrical property can be seen in two dimensions by plotting
  generated data-points:

  >>> mean = [0, 0]
  >>> cov = [[1, 0], [0, 100]]  # diagonal covariance

  Diagonal covariance means that points are oriented along x or y-axis:

  >>> import matplotlib.pyplot as plt
  >>> x, y = bm.random.multivariate_normal(mean, cov, 5000).T
  >>> plt.plot(x, y, 'x')
  >>> plt.axis('equal')
  >>> plt.show()

  Note that the covariance matrix must be positive semidefinite (a.k.a.
  nonnegative-definite). Otherwise, the behavior of this method is
  undefined and backwards compatibility is not guaranteed.

  References
  ----------
  .. [1] Papoulis, A., "Probability, Random Variables, and Stochastic
         Processes," 3rd ed., New York: McGraw-Hill, 1991.
  .. [2] Duda, R. O., Hart, P. E., and Stork, D. G., "Pattern
         Classification," 2nd ed., New York: Wiley, 2001.

  Examples
  --------
  >>> mean = (1, 2)
  >>> cov = [[1, 0], [0, 1]]
  >>> x = bm.random.multivariate_normal(mean, cov, (3, 3))
  >>> x.shape
  (3, 3, 2)

  Here we generate 800 samples from the bivariate normal distribution
  with mean [0, 0] and covariance matrix [[6, -3], [-3, 3.5]].  The
  expected variances of the first and second components of the sample
  are 6 and 3.5, respectively, and the expected correlation
  coefficient is -3/sqrt(6*3.5) ≈ -0.65465.

  >>> cov = np.array([[6, -3], [-3, 3.5]])
  >>> pts = bm.random.multivariate_normal([0, 0], cov, size=800)

  Check that the mean, covariance, and correlation coefficient of the
  sample are close to the expected values:

  >>> pts.mean(axis=0)
  array([ 0.0326911 , -0.01280782])  # may vary
  >>> np.cov(pts.T)
  array([[ 5.96202397, -2.85602287],
         [-2.85602287,  3.47613949]])  # may vary
  >>> np.corrcoef(pts.T)[0, 1]
  -0.6273591314603949  # may vary

  We can visualize this data with a scatter plot.  The orientation
  of the point cloud illustrates the negative correlation of the
  components of this sample.

  >>> import matplotlib.pyplot as plt
  >>> plt.plot(pts[:, 0], pts[:, 1], '.', alpha=0.5)
  >>> plt.axis('equal')
  >>> plt.grid()
  >>> plt.show()
  """
  return DEFAULT.multivariate_normal(mean, cov, size, method, key=key)


def negative_binomial(n, p, size: Optional[Union[int, Sequence[int]]] = None,
                      key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a negative binomial distribution.

  Samples are drawn from a negative binomial distribution with specified
  parameters, `n` successes and `p` probability of success where `n`
  is > 0 and `p` is in the interval [0, 1].

  Parameters
  ----------
  n : float or array_like of floats
      Parameter of the distribution, > 0.
  p : float or array_like of floats
      Parameter of the distribution, >= 0 and <=1.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``n`` and ``p`` are both scalars.
      Otherwise, ``np.broadcast(n, p).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized negative binomial distribution,
      where each sample is equal to N, the number of failures that
      occurred before a total of n successes was reached.

  Notes
  -----
  The probability mass function of the negative binomial distribution is

  .. math:: P(N;n,p) = \frac{\Gamma(N+n)}{N!\Gamma(n)}p^{n}(1-p)^{N},

  where :math:`n` is the number of successes, :math:`p` is the
  probability of success, :math:`N+n` is the number of trials, and
  :math:`\Gamma` is the gamma function. When :math:`n` is an integer,
  :math:`\frac{\Gamma(N+n)}{N!\Gamma(n)} = \binom{N+n-1}{N}`, which is
  the more common form of this term in the pmf. The negative
  binomial distribution gives the probability of N failures given n
  successes, with a success on the last trial.

  If one throws a die repeatedly until the third time a "1" appears,
  then the probability distribution of the number of non-"1"s that
  appear before the third "1" is a negative binomial distribution.

  References
  ----------
  .. [1] Weisstein, Eric W. "Negative Binomial Distribution." From
         MathWorld--A Wolfram Web Resource.
         http://mathworld.wolfram.com/NegativeBinomialDistribution.html
  .. [2] Wikipedia, "Negative binomial distribution",
         https://en.wikipedia.org/wiki/Negative_binomial_distribution

  Examples
  --------
  Draw samples from the distribution:

  A real world example. A company drills wild-cat oil
  exploration wells, each with an estimated probability of
  success of 0.1.  What is the probability of having one success
  for each successive well, that is what is the probability of a
  single success after drilling 5 wells, after 6 wells, etc.?

  >>> s = bm.random.negative_binomial(1, 0.1, 100000)
  >>> for i in range(1, 11): # doctest: +SKIP
  ...    probability = sum(s<i) / 100000.
  ...    print(i, "wells drilled, probability of one success =", probability)
  """
  return DEFAULT.negative_binomial(n, p, size, key=key)


def noncentral_chisquare(df, nonc, size: Optional[Union[int, Sequence[int]]] = None,
                         key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a noncentral chi-square distribution.

  The noncentral :math:`\chi^2` distribution is a generalization of
  the :math:`\chi^2` distribution.

  Parameters
  ----------
  df : float or array_like of floats
      Degrees of freedom, must be > 0.
  nonc : float or array_like of floats
      Non-centrality, must be non-negative.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``df`` and ``nonc`` are both scalars.
      Otherwise, ``np.broadcast(df, nonc).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized noncentral chi-square distribution.

  Notes
  -----
  The probability density function for the noncentral Chi-square
  distribution is

  .. math:: P(x;df,nonc) = \sum^{\infty}_{i=0}
                         \frac{e^{-nonc/2}(nonc/2)^{i}}{i!}
                         P_{Y_{df+2i}}(x),

  where :math:`Y_{q}` is the Chi-square with q degrees of freedom.

  References
  ----------
  .. [1] Wikipedia, "Noncentral chi-squared distribution"
         https://en.wikipedia.org/wiki/Noncentral_chi-squared_distribution

  Examples
  --------
  Draw values from the distribution and plot the histogram

  >>> import matplotlib.pyplot as plt
  >>> values = plt.hist(bm.random.noncentral_chisquare(3, 20, 100000),
  ...                   bins=200, density=True)
  >>> plt.show()

  Draw values from a noncentral chisquare with very small noncentrality,
  and compare to a chisquare.

  >>> plt.figure()
  >>> values = plt.hist(bm.random.noncentral_chisquare(3, .0000001, 100000),
  ...                   bins=np.arange(0., 25, .1), density=True)
  >>> values2 = plt.hist(bm.random.chisquare(3, 100000),
  ...                    bins=np.arange(0., 25, .1), density=True)
  >>> plt.plot(values[1][0:-1], values[0]-values2[0], 'ob')
  >>> plt.show()

  Demonstrate how large values of non-centrality lead to a more symmetric
  distribution.

  >>> plt.figure()
  >>> values = plt.hist(bm.random.noncentral_chisquare(3, 20, 100000),
  ...                   bins=200, density=True)
  >>> plt.show()
  """
  return DEFAULT.noncentral_chisquare(df, nonc, size, key=key)


def noncentral_f(dfnum, dfden, nonc, size: Optional[Union[int, Sequence[int]]] = None,
                 key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from the noncentral F distribution.

  Samples are drawn from an F distribution with specified parameters,
  `dfnum` (degrees of freedom in numerator) and `dfden` (degrees of
  freedom in denominator), where both parameters > 1.
  `nonc` is the non-centrality parameter.

  Parameters
  ----------
  dfnum : float or array_like of floats
      Numerator degrees of freedom, must be > 0.
  dfden : float or array_like of floats
      Denominator degrees of freedom, must be > 0.
  nonc : float or array_like of floats
      Non-centrality parameter, the sum of the squares of the numerator
      means, must be >= 0.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``dfnum``, ``dfden``, and ``nonc``
      are all scalars.  Otherwise, ``np.broadcast(dfnum, dfden, nonc).size``
      samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized noncentral Fisher distribution.

  Notes
  -----
  When calculating the power of an experiment (power = probability of
  rejecting the null hypothesis when a specific alternative is true) the
  non-central F statistic becomes important.  When the null hypothesis is
  true, the F statistic follows a central F distribution. When the null
  hypothesis is not true, then it follows a non-central F statistic.

  References
  ----------
  .. [1] Weisstein, Eric W. "Noncentral F-Distribution."
         From MathWorld--A Wolfram Web Resource.
         http://mathworld.wolfram.com/NoncentralF-Distribution.html
  .. [2] Wikipedia, "Noncentral F-distribution",
         https://en.wikipedia.org/wiki/Noncentral_F-distribution

  Examples
  --------
  In a study, testing for a specific alternative to the null hypothesis
  requires use of the Noncentral F distribution. We need to calculate the
  area in the tail of the distribution that exceeds the value of the F
  distribution for the null hypothesis.  We'll plot the two probability
  distributions for comparison.

  >>> dfnum = 3 # between group deg of freedom
  >>> dfden = 20 # within groups degrees of freedom
  >>> nonc = 3.0
  >>> nc_vals = bm.random.noncentral_f(dfnum, dfden, nonc, 1000000)
  >>> NF = np.histogram(nc_vals, bins=50, density=True)
  >>> c_vals = bm.random.f(dfnum, dfden, 1000000)
  >>> F = np.histogram(c_vals, bins=50, density=True)
  >>> import matplotlib.pyplot as plt
  >>> plt.plot(F[1][1:], F[0])
  >>> plt.plot(NF[1][1:], NF[0])
  >>> plt.show()
  """
  return DEFAULT.noncentral_f(dfnum, dfden, nonc, size, key=key)


def power(a,
          size: Optional[Union[int, Sequence[int]]] = None,
          key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draws samples in [0, 1] from a power distribution with positive
  exponent a - 1.

  Also known as the power function distribution.

  Parameters
  ----------
  a : float or array_like of floats
      Parameter of the distribution. Must be non-negative.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``a`` is a scalar.  Otherwise,
      ``np.array(a).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized power distribution.

  Raises
  ------
  ValueError
      If a <= 0.

  Notes
  -----
  The probability density function is

  .. math:: P(x; a) = ax^{a-1}, 0 \le x \le 1, a>0.

  The power function distribution is just the inverse of the Pareto
  distribution. It may also be seen as a special case of the Beta
  distribution.

  It is used, for example, in modeling the over-reporting of insurance
  claims.

  References
  ----------
  .. [1] Christian Kleiber, Samuel Kotz, "Statistical size distributions
         in economics and actuarial sciences", Wiley, 2003.
  .. [2] Heckert, N. A. and Filliben, James J. "NIST Handbook 148:
         Dataplot Reference Manual, Volume 2: Let Subcommands and Library
         Functions", National Institute of Standards and Technology
         Handbook Series, June 2003.
         https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/powpdf.pdf

  Examples
  --------
  Draw samples from the distribution:

  >>> a = 5. # shape
  >>> samples = 1000
  >>> s = bm.random.power(a, samples)

  Display the histogram of the samples, along with
  the probability density function:

  >>> import matplotlib.pyplot as plt
  >>> count, bins, ignored = plt.hist(s, bins=30)
  >>> x = np.linspace(0, 1, 100)
  >>> y = a*x**(a-1.)
  >>> normed_y = samples*np.diff(bins)[0]*y
  >>> plt.plot(x, normed_y)
  >>> plt.show()

  Compare the power function distribution to the inverse of the Pareto.

  >>> from scipy import stats # doctest: +SKIP
  >>> rvs = bm.random.power(5, 1000000)
  >>> rvsp = bm.random.pareto(5, 1000000)
  >>> xx = np.linspace(0,1,100)
  >>> powpdf = stats.powerlaw.pdf(xx,5)  # doctest: +SKIP

  >>> plt.figure()
  >>> plt.hist(rvs, bins=50, density=True)
  >>> plt.plot(xx,powpdf,'r-')  # doctest: +SKIP
  >>> plt.title('bm.random.power(5)')

  >>> plt.figure()
  >>> plt.hist(1./(1.+rvsp), bins=50, density=True)
  >>> plt.plot(xx,powpdf,'r-')  # doctest: +SKIP
  >>> plt.title('inverse of 1 + bm.random.pareto(5)')

  >>> plt.figure()
  >>> plt.hist(1./(1.+rvsp), bins=50, density=True)
  >>> plt.plot(xx,powpdf,'r-')  # doctest: +SKIP
  >>> plt.title('inverse of stats.pareto(5)')
  """
  return DEFAULT.power(a, size, key=key)


def rayleigh(scale=1.0,
             size: Optional[Union[int, Sequence[int]]] = None,
             key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a Rayleigh distribution.

  The :math:`\chi` and Weibull distributions are generalizations of the
  Rayleigh.

  Parameters
  ----------
  scale : float or array_like of floats, optional
      Scale, also equals the mode. Must be non-negative. Default is 1.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``scale`` is a scalar.  Otherwise,
      ``np.array(scale).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized Rayleigh distribution.

  Notes
  -----
  The probability density function for the Rayleigh distribution is

  .. math:: P(x;scale) = \frac{x}{scale^2}e^{\frac{-x^2}{2 \cdotp scale^2}}

  The Rayleigh distribution would arise, for example, if the East
  and North components of the wind velocity had identical zero-mean
  Gaussian distributions.  Then the wind speed would have a Rayleigh
  distribution.

  References
  ----------
  .. [1] Brighton Webs Ltd., "Rayleigh Distribution,"
         https://web.archive.org/web/20090514091424/http://brighton-webs.co.uk:80/distributions/rayleigh.asp
  .. [2] Wikipedia, "Rayleigh distribution"
         https://en.wikipedia.org/wiki/Rayleigh_distribution

  Examples
  --------
  Draw values from the distribution and plot the histogram

  >>> from matplotlib.pyplot import hist
  >>> values = hist(bm.random.rayleigh(3, 100000), bins=200, density=True)

  Wave heights tend to follow a Rayleigh distribution. If the mean wave
  height is 1 meter, what fraction of waves are likely to be larger than 3
  meters?

  >>> meanvalue = 1
  >>> modevalue = np.sqrt(2 / np.pi) * meanvalue
  >>> s = bm.random.rayleigh(modevalue, 1000000)

  The percentage of waves larger than 3 meters is:

  >>> 100.*sum(s>3)/1000000.
  0.087300000000000003 # random
  """
  return DEFAULT.rayleigh(scale, size, key=key)


def triangular(size: Optional[Union[int, Sequence[int]]] = None,
               key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from the triangular distribution over the
  interval ``[left, right]``.

  The triangular distribution is a continuous probability
  distribution with lower limit left, peak at mode, and upper
  limit right. Unlike the other distributions, these parameters
  directly define the shape of the pdf.

  Parameters
  ----------
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``left``, ``mode``, and ``right``
      are all scalars.  Otherwise, ``np.broadcast(left, mode, right).size``
      samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized triangular distribution.

  Notes
  -----
  The probability density function for the triangular distribution is

  .. math:: P(x;l, m, r) = \begin{cases}
            \frac{2(x-l)}{(r-l)(m-l)}& \text{for $l \leq x \leq m$},\\
            \frac{2(r-x)}{(r-l)(r-m)}& \text{for $m \leq x \leq r$},\\
            0& \text{otherwise}.
            \end{cases}

  The triangular distribution is often used in ill-defined
  problems where the underlying distribution is not known, but
  some knowledge of the limits and mode exists. Often it is used
  in simulations.

  References
  ----------
  .. [1] Wikipedia, "Triangular distribution"
         https://en.wikipedia.org/wiki/Triangular_distribution

  Examples
  --------
  Draw values from the distribution and plot the histogram:

  >>> import matplotlib.pyplot as plt
  >>> h = plt.hist(bm.random.triangular(-3, 0, 8, 100000), bins=200,
  ...              density=True)
  >>> plt.show()
  """
  return DEFAULT.triangular(size, key=key)


def vonmises(mu,
             kappa,
             size: Optional[Union[int, Sequence[int]]] = None,
             key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a von Mises distribution.

  Samples are drawn from a von Mises distribution with specified mode
  (mu) and dispersion (kappa), on the interval [-pi, pi].

  The von Mises distribution (also known as the circular normal
  distribution) is a continuous probability distribution on the unit
  circle.  It may be thought of as the circular analogue of the normal
  distribution.

  Parameters
  ----------
  mu : float or array_like of floats
      Mode ("center") of the distribution.
  kappa : float or array_like of floats
      Dispersion of the distribution, has to be >=0.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``mu`` and ``kappa`` are both scalars.
      Otherwise, ``np.broadcast(mu, kappa).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized von Mises distribution.

  See Also
  --------
  scipy.stats.vonmises : probability density function, distribution, or
      cumulative density function, etc.

  Notes
  -----
  The probability density for the von Mises distribution is

  .. math:: p(x) = \frac{e^{\kappa cos(x-\mu)}}{2\pi I_0(\kappa)},

  where :math:`\mu` is the mode and :math:`\kappa` the dispersion,
  and :math:`I_0(\kappa)` is the modified Bessel function of order 0.

  The von Mises is named for Richard Edler von Mises, who was born in
  Austria-Hungary, in what is now the Ukraine.  He fled to the United
  States in 1939 and became a professor at Harvard.  He worked in
  probability theory, aerodynamics, fluid mechanics, and philosophy of
  science.

  References
  ----------
  .. [1] Abramowitz, M. and Stegun, I. A. (Eds.). "Handbook of
         Mathematical Functions with Formulas, Graphs, and Mathematical
         Tables, 9th printing," New York: Dover, 1972.
  .. [2] von Mises, R., "Mathematical Theory of Probability
         and Statistics", New York: Academic Press, 1964.

  Examples
  --------
  Draw samples from the distribution:

  >>> mu, kappa = 0.0, 4.0 # mean and dispersion
  >>> s = bm.random.vonmises(mu, kappa, 1000)

  Display the histogram of the samples, along with
  the probability density function:

  >>> import matplotlib.pyplot as plt
  >>> from scipy.special import i0  # doctest: +SKIP
  >>> plt.hist(s, 50, density=True)
  >>> x = np.linspace(-np.pi, np.pi, num=51)
  >>> y = np.exp(kappa*np.cos(x-mu))/(2*np.pi*i0(kappa))  # doctest: +SKIP
  >>> plt.plot(x, y, linewidth=2, color='r')  # doctest: +SKIP
  >>> plt.show()
  """
  return DEFAULT.vonmises(mu, kappa, size, key=key)


def wald(mean,
         scale,
         size: Optional[Union[int, Sequence[int]]] = None,
         key: Optional[Union[int, JAX_RAND_KEY]] = None):
  r"""
  Draw samples from a Wald, or inverse Gaussian, distribution.

  As the scale approaches infinity, the distribution becomes more like a
  Gaussian. Some references claim that the Wald is an inverse Gaussian
  with mean equal to 1, but this is by no means universal.

  The inverse Gaussian distribution was first studied in relationship to
  Brownian motion. In 1956 M.C.K. Tweedie used the name inverse Gaussian
  because there is an inverse relationship between the time to cover a
  unit distance and distance covered in unit time.

  Parameters
  ----------
  mean : float or array_like of floats
      Distribution mean, must be > 0.
  scale : float or array_like of floats
      Scale parameter, must be > 0.
  size : int or tuple of ints, optional
      Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
      ``m * n * k`` samples are drawn.  If size is ``None`` (default),
      a single value is returned if ``mean`` and ``scale`` are both scalars.
      Otherwise, ``np.broadcast(mean, scale).size`` samples are drawn.

  Returns
  -------
  out : ndarray or scalar
      Drawn samples from the parameterized Wald distribution.

  Notes
  -----
  The probability density function for the Wald distribution is

  .. math:: P(x;mean,scale) = \sqrt{\frac{scale}{2\pi x^3}}e^
                              \frac{-scale(x-mean)^2}{2\cdotp mean^2x}

  As noted above the inverse Gaussian distribution first arise
  from attempts to model Brownian motion. It is also a
  competitor to the Weibull for use in reliability modeling and
  modeling stock returns and interest rate processes.

  References
  ----------
  .. [1] Brighton Webs Ltd., Wald Distribution,
         https://web.archive.org/web/20090423014010/http://www.brighton-webs.co.uk:80/distributions/wald.asp
  .. [2] Chhikara, Raj S., and Folks, J. Leroy, "The Inverse Gaussian
         Distribution: Theory : Methodology, and Applications", CRC Press,
         1988.
  .. [3] Wikipedia, "Inverse Gaussian distribution"
         https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution

  Examples
  --------
  Draw values from the distribution and plot the histogram:

  >>> import matplotlib.pyplot as plt
  >>> h = plt.hist(bm.random.wald(3, 2, 100000), bins=200, density=True)
  >>> plt.show()
  """
  return DEFAULT.wald(mean, scale, size, key=key)


def weibull(a,
            size: Optional[Union[int, Sequence[int]]] = None,
            key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def weibull_min(a,
                scale=None,
                size: Optional[Union[int, Sequence[int]]] = None,
                key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def zipf(a,
         size: Optional[Union[int, Sequence[int]]] = None,
         key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def maxwell(size: Optional[Union[int, Sequence[int]]] = None,
            key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def t(df,
      size: Optional[Union[int, Sequence[int]]] = None,
      key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def orthogonal(n: int,
               size: Optional[Union[int, Sequence[int]]] = None,
               key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def loggamma(a,
             size: Optional[Union[int, Sequence[int]]] = None,
             key: Optional[Union[int, JAX_RAND_KEY]] = None):
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
  return DEFAULT.loggamma(a, size, key=key)


def categorical(logits,
                axis: int = -1,
                size: Optional[Union[int, Sequence[int]]] = None,
                key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def rand_like(input, *, dtype=None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def randn_like(input, *, dtype=None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
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


def randint_like(input, low=0, high=None, *, dtype=None, key: Optional[Union[int, JAX_RAND_KEY]] = None):
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
