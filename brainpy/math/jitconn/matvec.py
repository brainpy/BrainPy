# -*- coding: utf-8 -*-
# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Tuple, Optional, Union

import brainevent
import jax
import numpy as np

from brainpy.math.ndarray import Array as Array

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

    Parameters::

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

    Returns::

    out: Array, ndarray
      The output of :math:`y = M @ v`.
    """
    if seed is None:
        seed = np.random.randint(0, 1000000000)
    if isinstance(vector, Array):
        vector = vector.value
    if isinstance(weight, Array):
        weight = weight.value

    csr = brainevent.JITCHomoR((weight, conn_prob, seed), shape=shape, corder=outdim_parallel)
    if transpose:
        return vector @ csr
    else:
        return csr @ vector


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

    Parameters::

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

    Returns::

    out: Array, ndarray
      The output of :math:`y = M @ v`.
    """
    if seed is None:
        seed = np.random.randint(0, 1000000000)
    if isinstance(vector, Array):
        vector = vector.value
    if isinstance(w_low, Array):
        w_low = w_low.value
    if isinstance(w_high, Array):
        w_high = w_high.value

    csr = brainevent.JITCUniformR((w_low, w_high, conn_prob, seed), shape=shape, corder=outdim_parallel)
    if transpose:
        return vector @ csr
    else:
        return csr @ vector


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

    Parameters::

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

    Returns::

    out: Array, ndarray
      The output of :math:`y = M @ v`.
    """
    if seed is None:
        seed = np.random.randint(0, 1000000000)
    if isinstance(vector, Array):
        vector = vector.value
    if isinstance(w_mu, Array):
        w_mu = w_mu.value
    if isinstance(w_sigma, Array):
        w_sigma = w_sigma.value

    csr = brainevent.JITCNormalR((w_mu, w_sigma, conn_prob, seed), shape=shape, corder=outdim_parallel)
    if transpose:
        return vector @ csr
    else:
        return csr @ vector


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

    Parameters::

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

    Returns::

    out: Array, ndarray
      The connection matrix :math:`M`.
    """
    if seed is None:
        seed = np.random.randint(0, 1000000000)
    csr = brainevent.JITCHomoR((weight, conn_prob, seed), shape=shape, corder=outdim_parallel)
    if transpose:
        csr = csr.T
    return csr.todense()


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

    Parameters::

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

    Returns::

    out: Array, ndarray
      The weight matrix :math:`M`.
    """
    if seed is None:
        seed = np.random.randint(0, 1000000000)
    if isinstance(w_low, Array):
        w_low = w_low.value
    if isinstance(w_high, Array):
        w_high = w_high.value

    csr = brainevent.JITCUniformR((w_low, w_high, conn_prob, seed), shape=shape, corder=outdim_parallel)
    if transpose:
        csr = csr.T
    return csr.todense()


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

    Parameters::

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

    Returns::

    out: Array, ndarray
      The weight matrix :math:`M`.
    """
    if seed is None:
        seed = np.random.randint(0, 1000000000)
    if isinstance(w_mu, Array):
        w_mu = w_mu.value
    if isinstance(w_sigma, Array):
        w_sigma = w_sigma.value

    csr = brainevent.JITCNormalR((w_mu, w_sigma, conn_prob, seed), shape=shape, corder=outdim_parallel)
    if transpose:
        csr = csr.T
    return csr.todense()
