# -*- coding: utf-8 -*-


from typing import Tuple, Optional

import jax
from jax import numpy as jnp

from brainpy._src import tools

__all__ = [
  'event_matvec_prob_conn_homo_weight',
  'event_matvec_prob_conn_uniform_weight',
  'event_matvec_prob_conn_normal_weight',

  'matmat_prob_conn_homo_weight',
  'matmat_prob_conn_uniform_weight',
  'matmat_prob_conn_normal_weight',

  'matvec_prob_conn_homo_weight',
  'matvec_prob_conn_uniform_weight',
  'matvec_prob_conn_normal_weight',
]


def matvec_prob_conn_homo_weight(
    vector: jnp.ndarray,
    weight: float,
    *,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
    outdim_parallel: bool = True,
    version: str = 'v2'
) -> jnp.ndarray:
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
  bl = tools.import_brainpylib()
  return bl.jitconn_ops.matvec_prob_conn_homo_weight(vector,
                                                     weight,
                                                     conn_prob=conn_prob,
                                                     shape=shape,
                                                     seed=seed,
                                                     transpose=transpose,
                                                     outdim_parallel=outdim_parallel,
                                                     version=version)


def matvec_prob_conn_uniform_weight(
    vector: jnp.ndarray,
    *,
    w_low: float,
    w_high: float,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
    outdim_parallel: bool = True,
    version: str = 'v2'
) -> jnp.ndarray:
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
  bl = tools.import_brainpylib()
  return bl.jitconn_ops.matvec_prob_conn_uniform_weight(vector,
                                                        w_low=w_low,
                                                        w_high=w_high,
                                                        conn_prob=conn_prob,
                                                        shape=shape,
                                                        seed=seed,
                                                        transpose=transpose,
                                                        outdim_parallel=outdim_parallel,
                                                        version=version)


def matvec_prob_conn_normal_weight(
    vector: jnp.ndarray,
    *,
    w_mu: float,
    w_sigma: float,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
    outdim_parallel: bool = True,
    version: str = 'v2'
) -> jnp.ndarray:
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
  bl = tools.import_brainpylib()
  return bl.jitconn_ops.matvec_prob_conn_normal_weight(vector,
                                                       w_mu=w_mu,
                                                       w_sigma=w_sigma,
                                                       conn_prob=conn_prob,
                                                       shape=shape,
                                                       seed=seed,
                                                       transpose=transpose,
                                                       outdim_parallel=outdim_parallel,
                                                       version=version)


def event_matvec_prob_conn_homo_weight(
    events: jnp.ndarray,
    weight: float,
    *,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jnp.ndarray:
  bl = tools.import_brainpylib()
  return bl.jitconn_ops.event_matvec_prob_conn_homo_weight(events, weight,
                                                           conn_prob=conn_prob,
                                                           shape=shape,
                                                           seed=seed,
                                                           transpose=transpose,
                                                           outdim_parallel=outdim_parallel)


event_matvec_prob_conn_homo_weight.__doc__ = matvec_prob_conn_homo_weight.__doc__


def event_matvec_prob_conn_uniform_weight(
    events: jnp.ndarray,
    *,
    w_low: float,
    w_high: float,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jnp.ndarray:
  bl = tools.import_brainpylib()
  return bl.jitconn_ops.event_matvec_prob_conn_uniform_weight(events,
                                                              w_low=w_low,
                                                              w_high=w_high,
                                                              conn_prob=conn_prob,
                                                              shape=shape,
                                                              seed=seed,
                                                              transpose=transpose,
                                                              outdim_parallel=outdim_parallel)[0]


event_matvec_prob_conn_uniform_weight.__doc__ = matvec_prob_conn_uniform_weight.__doc__


def event_matvec_prob_conn_normal_weight(
    events: jnp.ndarray,
    *,
    w_mu: float,
    w_sigma: float,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jnp.ndarray:
  bl = tools.import_brainpylib()
  return bl.jitconn_ops.event_matvec_prob_conn_normal_weight(events,
                                                             w_mu=w_mu,
                                                             w_sigma=w_sigma,
                                                             conn_prob=conn_prob,
                                                             shape=shape,
                                                             seed=seed,
                                                             transpose=transpose,
                                                             outdim_parallel=outdim_parallel)[0]


event_matvec_prob_conn_normal_weight.__doc__ = matvec_prob_conn_normal_weight.__doc__


def matmat_prob_conn_homo_weight(
    vector: jnp.ndarray,
    weight: float,
    *,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jnp.ndarray:
  r"""Perform the :math:`Y=X@M` operation, where :math:`X`, :math:`Y` and :math:`M` are matrices,
  and :math:`M` is just-in-time randomly generated with a scalar `weight` at each position.

  This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
  on CPU and GPU devices.

  .. warning::

     This API may change in the future.

  In this operation, :math:`M` is the random matrix with a connection probability
  `conn_prob`, and at each connection the value is the same scalar `weight`.

  When ``transpose=True``, we perform an operation of :math:`Y=X@M^T`.

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
    The output of :math:`Y = X @ M`.
  """
  bl = tools.import_brainpylib()
  return bl.jitconn_ops.matmat_prob_conn_homo_weight(vector,
                                                     weight,
                                                     conn_prob=conn_prob,
                                                     shape=shape,
                                                     seed=seed,
                                                     transpose=transpose,
                                                     outdim_parallel=outdim_parallel)


def matmat_prob_conn_uniform_weight(
    matrix: jax.Array,
    *,
    w_low: float,
    w_high: float,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    version: str = 'v1'
) -> jnp.ndarray:
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
  matrix: Array
    The matrix :math:`X`.
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

  Returns
  -------
  out: Array, ndarray
    The output of :math:`y = M @ v`.
  """
  bl = tools.import_brainpylib()
  return bl.jitconn_ops.matmat_prob_conn_uniform_weight(matrix,
                                                        w_low=w_low,
                                                        w_high=w_high,
                                                        conn_prob=conn_prob,
                                                        shape=shape,
                                                        seed=seed,
                                                        version=version)


def matmat_prob_conn_normal_weight(
    matrix: jax.Array,
    *,
    w_mu: float,
    w_sigma: float,
    conn_prob: float,
    shape: Tuple[int, int],
    seed: Optional[int] = None,
    version: str = 'v1'
) -> jax.Array:
  r"""Perform the :math:`Y=X@M` operation, where :math:`X`, :math:`Y` and :math:`M` are matrices,
  and :math:`M` is just-in-time randomly generated with a normal distribution for its value.

  This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
  on CPU and GPU devices.

  .. warning::

     This API may change in the future.

  In this operation, :math:`M` is the random matrix with a connection probability
  `conn_prob`, and at each connection the value is the same scalar `weight`.

  Parameters
  ----------
  matrix: Array
    The matrix :math:`X`.
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

  Returns
  -------
  out: Array, ndarray
    The output of :math:`y = M @ v`.
  """
  bl = tools.import_brainpylib()
  return bl.jitconn_ops.matmat_prob_conn_normal_weight(matrix,
                                                       w_mu=w_mu,
                                                       w_sigma=w_sigma,
                                                       conn_prob=conn_prob,
                                                       shape=shape,
                                                       seed=seed,
                                                       version=version)
