# -*- coding: utf-8 -*-
import numbers
from typing import Tuple, Optional, Union

import jax
import numpy as np
from braintaichi import jitc_mv_prob_homo, jitc_mv_prob_uniform
from jax import numpy as jnp

from brainpy._src.dependency_check import import_taichi
from brainpy._src.math import defaults
from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import Array
from brainpy._src.math.op_register import XLACustomOp
from brainpy.errors import PackageMissingError

ti = import_taichi(error_if_not_found=False)

__all__ = [
  'mv_prob_homo',
  'mv_prob_uniform',
  'mv_prob_normal',
  'get_homo_weight_matrix',
  'get_uniform_weight_matrix',
  'get_normal_weight_matrix'
]


def mv_prob_homo(
    vector: Union[Array, jax.Array],
    weight: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  r"""Perform the :math:`y=M@v` operation,
  where :math:`M` is just-in-time randomly generated with a scalar `weight` at each position.

  This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
  on CPU and GPU devices.

  .. warning::

     This API may change in the future.

  In this operation, :math:`M` is the random matrix with a connection probability
  `conn_prob`, and at each connection the value is the same scalar `weight`.

  When ``transpose=True``, we perform an operation of :math:`y=M^T@v`.

  .. note::

     Note that the just-in-time generated :math:`M` (`transpose=False`) is
     different from the generated :math:`M^T` (`transpose=True`).

     If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
     matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
     the speed compared with ``outdim_parallel=False``.

  Parameters
  ----------
  vector: Array, ndarray
    The vector.
  weight: float
    The value of the random matrix.
  conn_prob: float
    The connection probability.
  shape: tuple of int
    The matrix shape.
  seed: int
    The random number generation seed.
  transpose: bool
    Transpose the random matrix or not.
  outdim_parallel: bool
    Perform the parallel random generations along the out dimension or not.
    It can be used to set the just-in-time generated :math:M^T: is the same
    as the just-in-time generated :math:`M` when ``transpose=True``.

  Returns
  -------
  out: Array, ndarray
    The output of :math:`y = M @ v`.
  """
  return jitc_mv_prob_homo(vector, weight, conn_prob, seed, shape=shape,
                          transpose=transpose, outdim_parallel=outdim_parallel)


def mv_prob_uniform(
    vector: jax.Array,
    w_low: float,
    w_high: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  r"""Perform the :math:`y=M@v` operation,
  where :math:`M` is just-in-time randomly generated with a uniform distribution for its value.

  This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
  on CPU and GPU devices.

  .. warning::

     This API may change in the future.

  In this operation, :math:`M` is the random matrix with a connection probability
  `conn_prob`, and at each connection the value is the same scalar `weight`.

  When ``transpose=True``, we perform an operation of :math:`y=M^T@v`.

  .. note::

     Note that the just-in-time generated :math:`M` (`transpose=False`) is
     different from the generated :math:`M^T` (`transpose=True`).

     If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
     matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
     the speed compared with ``outdim_parallel=False``.

  Parameters
  ----------
  vector: Array, ndarray
    The vector.
  w_low: float
    Lower boundary of the output interval.
  w_high: float
    Upper boundary of the output interval.
  conn_prob: float
    The connection probability.
  shape: tuple of int
    The matrix shape.
  seed: int
    The random number generation seed.
  transpose: bool
    Transpose the random matrix or not.
  outdim_parallel: bool
    Perform the parallel random generations along the out dimension or not.
    It can be used to set the just-in-time generated :math:M^T: is the same
    as the just-in-time generated :math:`M` when ``transpose=True``.

  Returns
  -------
  out: Array, ndarray
    The output of :math:`y = M @ v`.
  """
  return jitc_mv_prob_uniform(vector, w_low, w_high, conn_prob, seed, shape=shape,
                             transpose=transpose, outdim_parallel=outdim_parallel)


def mv_prob_normal(
    vector: jax.Array,
    w_mu: float,
    w_sigma: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  r"""Perform the :math:`y=M@v` operation,
  where :math:`M` is just-in-time randomly generated with a normal distribution for its value.

  This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
  on CPU and GPU devices.

  .. warning::

     This API may change in the future.

  In this operation, :math:`M` is the random matrix with a connection probability
  `conn_prob`, and at each connection the value is the same scalar `weight`.

  When ``transpose=True``, we perform an operation of :math:`y=M^T@v`.

  .. note::

     Note that the just-in-time generated :math:`M` (`transpose=False`) is
     different from the generated :math:`M^T` (`transpose=True`).

     If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
     matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
     the speed compared with ``outdim_parallel=False``.

  Parameters
  ----------
  vector: Array, ndarray
    The vector.
  w_mu: float
    Mean (centre) of the distribution.
  w_sigma: float
    Standard deviation (spread or “width”) of the distribution. Must be non-negative.
  conn_prob: float
    The connection probability.
  shape: tuple of int
    The matrix shape.
  seed: int
    The random number generation seed.
  transpose: bool
    Transpose the random matrix or not.
  outdim_parallel: bool
    Perform the parallel random generations along the out dimension or not.
    It can be used to set the just-in-time generated :math:M^T: is the same
    as the just-in-time generated :math:`M` when ``transpose=True``.

  Returns
  -------
  out: Array, ndarray
    The output of :math:`y = M @ v`.
  """
  return jitc_mv_prob_uniform(vector, w_mu, w_sigma, conn_prob, seed, shape=shape,
                            transpose=transpose, outdim_parallel=outdim_parallel)


def get_homo_weight_matrix(
    weight: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  r"""Get the connection matrix :math:`M` with a connection probability `conn_prob`.

  Parameters
  ----------
  conn_prob: float
    The connection probability.
  shape: tuple of int
    The matrix shape.
  seed: int
    The random number generation seed.
  transpose: bool
    Transpose the random matrix or not.
  outdim_parallel: bool
    Perform the parallel random generations along the out dimension or not.
    It can be used to set the just-in-time generated :math:M^T: is the same
    as the just-in-time generated :math:`M` when ``transpose=True``.

  Returns
  -------
  out: Array, ndarray
    The connection matrix :math:`M`.
  """
  if isinstance(weight, numbers.Number):
    weight = jnp.atleast_1d(jnp.asarray(weight, dtype=defaults.float_))
  else:
    raise ValueError(f'weight must be a number type, but get {type(weight)}')
  if ti is None:
    raise PackageMissingError.by_purpose('taichi', purpose='customized operators')

  conn_len = jnp.ceil(1 / conn_prob) * 2 - 1
  conn_len = jnp.asarray(jnp.atleast_1d(conn_len), dtype=jnp.int32)
  if seed is None:
    with jax.ensure_compile_time_eval():
      seed = np.random.randint(0, int(1e8), 1)
  seed = jnp.atleast_1d(jnp.asarray(seed, dtype=jnp.uint32))
  r = raw_get_homo_weight_matrix(conn_len, seed, shape=shape,
                                 transpose=transpose, outdim_parallel=outdim_parallel)[0].astype(jnp.bool_)
  r *= weight
  if transpose:
    return r.transpose()
  else:
    return r


def get_uniform_weight_matrix(
    w_low: float,
    w_high: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  r"""Get the weight matrix :math:`M` with a uniform distribution for its value.

  Parameters
  ----------
  w_low: float
    Lower boundary of the output interval.
  w_high: float
    Upper boundary of the output interval.
  conn_prob: float
    The connection probability.
  shape: tuple of int
    The matrix shape.
  seed: int
    The random number generation seed.
  transpose: bool
    Transpose the random matrix or not.
  outdim_parallel: bool
    Perform the parallel random generations along the out dimension or not.
    It can be used to set the just-in-time generated :math:M^T: is the same
    as the just-in-time generated :math:`M` when ``transpose=True``.

  Returns
  -------
  out: Array, ndarray
    The weight matrix :math:`M`.
  """
  if ti is None:
    raise PackageMissingError.by_purpose('taichi', purpose='customized operators')

  w_low = jnp.atleast_1d(as_jax(w_low))
  w_high = jnp.atleast_1d(as_jax(w_high))
  conn_len = jnp.ceil(1 / conn_prob) * 2 - 1
  conn_len = jnp.asarray(jnp.atleast_1d(conn_len), dtype=jnp.int32)
  if seed is None:
    with jax.ensure_compile_time_eval():
      seed = np.random.randint(0, int(1e8), 1)
  seed = jnp.atleast_1d(jnp.asarray(seed, dtype=jnp.uint32))
  r = raw_get_uniform_weight_matrix(w_low, w_high, conn_len, seed, shape=shape,
                                    transpose=transpose, outdim_parallel=outdim_parallel)[0]
  if transpose:
    return r.transpose()
  else:
    return r


def get_normal_weight_matrix(
    w_mu: float,
    w_sigma: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  r"""Get the weight matrix :math:`M` with a normal distribution for its value.

  Parameters
  ----------
  w_mu: float
    Mean (centre) of the distribution.
  w_sigma: float
    Standard deviation (spread or “width”) of the distribution. Must be non-negative.
  shape: tuple of int
    The matrix shape.
  seed: int
    The random number generation seed.
  transpose: bool
    Transpose the random matrix or not.
  outdim_parallel: bool
    Perform the parallel random generations along the out dimension or not.
    It can be used to set the just-in-time generated :math:M^T: is the same
    as the just-in-time generated :math:`M` when ``transpose=True``.

  Returns
  -------
  out: Array, ndarray
    The weight matrix :math:`M`.
  """
  if ti is None:
    raise PackageMissingError.by_purpose('taichi', purpose='customized operators')

  w_mu = jnp.atleast_1d(as_jax(w_mu))
  w_sigma = jnp.atleast_1d(as_jax(w_sigma))
  conn_len = jnp.ceil(1 / conn_prob) * 2 - 1
  conn_len = jnp.asarray(jnp.atleast_1d(conn_len), dtype=jnp.int32)
  if seed is None:
    with jax.ensure_compile_time_eval():
      seed = np.random.randint(0, int(1e8), 1)
  seed = jnp.atleast_1d(jnp.asarray(seed, dtype=jnp.uint32))
  r = raw_get_normal_weight_matrix(w_mu, w_sigma, conn_len, seed,
                                   shape=shape,
                                   transpose=transpose, outdim_parallel=outdim_parallel)[0]
  if transpose:
    return r.transpose()
  else:
    return r


def raw_get_homo_weight_matrix(
    conn_len: jax.Array,
    seed: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  if outdim_parallel:
    prim = _get_connect_matrix_outdim_parallel_p
  else:
    prim = _get_connect_matrix_p

  return prim(conn_len,
              seed,
              outs=[jax.ShapeDtypeStruct(shape=shape, dtype=jnp.int32)],
              shape=shape,
              transpose=transpose,
              outdim_parallel=outdim_parallel)


def raw_get_uniform_weight_matrix(
    w_low: jax.Array,
    w_high: jax.Array,
    conn_len: jax.Array,
    seed: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  if outdim_parallel:
    prim = _get_uniform_weight_matrix_outdim_parallel_p
  else:
    prim = _get_uniform_weight_matrix_p

  return prim(w_low,
              w_high,
              conn_len,
              seed,
              outs=[jax.ShapeDtypeStruct(shape=shape, dtype=jnp.float32)],
              shape=shape,
              transpose=transpose,
              outdim_parallel=outdim_parallel)


def raw_get_normal_weight_matrix(
    w_mu: jax.Array,
    w_sigma: jax.Array,
    conn_len: jax.Array,
    seed: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  if outdim_parallel:
    prim = _get_normal_weight_matrix_outdim_parallel_p
  else:
    prim = _get_normal_weight_matrix_p

  return prim(w_mu,
              w_sigma,
              conn_len,
              seed,
              outs=[jax.ShapeDtypeStruct(shape=shape, dtype=jnp.float32)],
              shape=shape,
              transpose=transpose,
              outdim_parallel=outdim_parallel)



if ti is not None:
  from brainpy._src.math.tifunc import (lfsr88_key, lfsr88_random_integers, lfsr88_uniform, lfsr88_normal)

  @ti.kernel
  def _get_connect_matrix(
      clen: ti.types.ndarray(),
      seed: ti.types.ndarray(),
      out: ti.types.ndarray(),
  ):
    num_row = out.shape[0]
    num_col = out.shape[1]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_col in range(num_col):
      key = lfsr88_key(seed0 + i_col)
      key, i_row = lfsr88_random_integers(key, 0, clen0 - 1)
      while i_row < num_row:
        out[i_row, i_col] = 1
        key, inc = lfsr88_random_integers(key, 1, clen0)
        i_row += inc


  @ti.kernel
  def _get_connect_matrix_outdim_parallel(
      clen: ti.types.ndarray(),
      seed: ti.types.ndarray(),
      out: ti.types.ndarray(),
  ):
    num_row = out.shape[0]
    num_col = out.shape[1]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_row in range(num_row):
      key = lfsr88_key(seed0 + i_row)
      key, i_col = lfsr88_random_integers(key, 0, clen0 - 1)
      while i_col < num_col:
        out[i_row, i_col] = 1
        key, inc = lfsr88_random_integers(key, 1, clen0)
        i_col += inc


  _get_connect_matrix_p = XLACustomOp(cpu_kernel=_get_connect_matrix, gpu_kernel=_get_connect_matrix)
  _get_connect_matrix_outdim_parallel_p = XLACustomOp(cpu_kernel=_get_connect_matrix_outdim_parallel,
                                                      gpu_kernel=_get_connect_matrix_outdim_parallel)


  @ti.kernel
  def _get_uniform_weight_matrix(
      w_low: ti.types.ndarray(),
      w_high: ti.types.ndarray(),
      clen: ti.types.ndarray(),
      seed: ti.types.ndarray(),
      out: ti.types.ndarray(),
  ):
    num_row = out.shape[0]
    num_col = out.shape[1]
    w_low0 = w_low[0]
    w_high0 = w_high[0]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_col in range(num_col):
      key = lfsr88_key(seed0 + i_col)
      key, i_row = lfsr88_random_integers(key, 0, clen0 - 1)
      while i_row < num_row:
        key, raw_v = lfsr88_uniform(key, w_low0, w_high0)
        out[i_row, i_col] = raw_v
        key, inc = lfsr88_random_integers(key, 1, clen0)
        i_row += inc


  @ti.kernel
  def _get_uniform_weight_matrix_outdim_parallel(
      w_low: ti.types.ndarray(),
      w_high: ti.types.ndarray(),
      clen: ti.types.ndarray(),
      seed: ti.types.ndarray(),
      out: ti.types.ndarray(),
  ):
    num_row = out.shape[0]
    num_col = out.shape[1]
    w_low0 = w_low[0]
    w_high0 = w_high[0]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_row in range(num_row):
      key = lfsr88_key(seed0 + i_row)
      key, i_col = lfsr88_random_integers(key, 0, clen0 - 1)
      while i_col < num_col:
        key, raw_v = lfsr88_uniform(key, w_low0, w_high0)
        out[i_row, i_col] = raw_v
        key, inc = lfsr88_random_integers(key, 1, clen0)
        i_col += inc


  _get_uniform_weight_matrix_p = XLACustomOp(cpu_kernel=_get_uniform_weight_matrix,
                                             gpu_kernel=_get_uniform_weight_matrix)
  _get_uniform_weight_matrix_outdim_parallel_p = XLACustomOp(cpu_kernel=_get_uniform_weight_matrix_outdim_parallel,
                                                             gpu_kernel=_get_uniform_weight_matrix_outdim_parallel)


  @ti.kernel
  def _get_normal_weight_matrix(
      w_mu: ti.types.ndarray(),
      w_sigma: ti.types.ndarray(),
      clen: ti.types.ndarray(),
      seed: ti.types.ndarray(),
      out: ti.types.ndarray(),
  ):
    num_row = out.shape[0]
    num_col = out.shape[1]
    w_mu0 = w_mu[0]
    w_sigma0 = w_sigma[0]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_col in range(num_col):
      key = lfsr88_key(seed0 + i_col)
      key, i_row = lfsr88_random_integers(key, 0, clen0 - 1)
      while i_row < num_row:
        key, raw_v = lfsr88_normal(key, w_mu0, w_sigma0)
        out[i_row, i_col] = raw_v
        key, inc = lfsr88_random_integers(key, 1, clen0)
        i_row += inc


  @ti.kernel
  def _get_normal_weight_matrix_outdim_parallel(
      w_mu: ti.types.ndarray(),
      w_sigma: ti.types.ndarray(),
      clen: ti.types.ndarray(),
      seed: ti.types.ndarray(),
      out: ti.types.ndarray(),
  ):
    num_row = out.shape[0]
    num_col = out.shape[1]
    w_mu0 = w_mu[0]
    w_sigma0 = w_sigma[0]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_row in range(num_row):
      key = lfsr88_key(seed0 + i_row)
      key, i_col = lfsr88_random_integers(key, 0, clen0 - 1)
      while i_col < num_col:
        key, raw_v = lfsr88_normal(key, w_mu0, w_sigma0)
        out[i_row, i_col] = raw_v
        key, inc = lfsr88_random_integers(key, 1, clen0)
        i_col += inc


  _get_normal_weight_matrix_p = XLACustomOp(cpu_kernel=_get_normal_weight_matrix,
                                            gpu_kernel=_get_normal_weight_matrix)
  _get_normal_weight_matrix_outdim_parallel_p = XLACustomOp(cpu_kernel=_get_normal_weight_matrix_outdim_parallel,
                                                            gpu_kernel=_get_normal_weight_matrix_outdim_parallel)
