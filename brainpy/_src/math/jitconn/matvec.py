# -*- coding: utf-8 -*-
from typing import Tuple, Optional, Union

import jax

from brainpy._src.dependency_check import import_braintaichi, raise_braintaichi_not_found
from brainpy._src.math.ndarray import Array

bti = import_braintaichi(error_if_not_found=False)

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
    if bti is None:
        raise_braintaichi_not_found()

    return bti.jitc_mv_prob_homo(vector, weight, conn_prob, seed, shape=shape,
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
    if bti is None:
        raise_braintaichi_not_found()

    return bti.jitc_mv_prob_uniform(vector, w_low, w_high, conn_prob, seed, shape=shape,
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
    if bti is None:
        raise_braintaichi_not_found()
    return bti.jitc_mv_prob_normal(vector, w_mu, w_sigma, conn_prob, seed, shape=shape,
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
    if bti is None:
        raise_braintaichi_not_found()
    return bti.get_homo_weight_matrix(weight, conn_prob, seed, shape=shape, transpose=transpose,
                                      outdim_parallel=outdim_parallel)


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
    if bti is None:
        raise_braintaichi_not_found()
    return bti.get_uniform_weight_matrix(w_low, w_high, conn_prob, seed, shape=shape,
                                         transpose=transpose, outdim_parallel=outdim_parallel)


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
    if bti is None:
        raise_braintaichi_not_found()
    return bti.get_normal_weight_matrix(w_mu, w_sigma, conn_prob, seed,
                                        shape=shape,
                                        transpose=transpose, outdim_parallel=outdim_parallel)
