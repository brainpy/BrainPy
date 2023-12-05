# -*- coding: utf-8 -*-


from functools import partial
from typing import Tuple, Optional, Union

import jax
import numpy as np
import taichi as ti
from jax import numpy as jnp, dtypes
from jax.core import ShapedArray, Primitive
from jax.interpreters import xla, ad
from jax.lib import xla_client

from brainpy._src.dependency_check import import_brainpylib_gpu_ops, import_brainpylib_cpu_ops
from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import Array, _get_dtype
from brainpy._src.math.op_register import XLACustomOp
from brainpy._src.math.random import (taichi_lcg_rand as random_generator,
                                      taichi_uniform_int_distribution as uniform_int_distribution,
                                      taichi_uniform_real_distribution as uniform_real_distribution,
                                      taichi_normal_distribution as normal_distribution,)
from brainpy.errors import GPUOperatorNotFound

__all__ = [
  'mv_prob_homo_taichi',
  'mv_prob_uniform_taichi',
  'mv_prob_normal_taichi',
]

@ti.func
def _dist1(seed: ti.types.ndarray(ndim=1),
           clen: ti.types.ndarray(ndim=1)
):
    return uniform_int_distribution(random_generator(seed), 1, clen[0])

@ti.kernel
def _mv_prob_homo_outdim_parallel_cpu(
    vector: ti.types.ndarray(ndim=1),
    weight: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    shape: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = shape[0]
    num_col = shape[1]
    weight_value = weight[0]
    ti.loop_config(serialize=True)
    for i_col in range(num_col):
        i_row = _dist1(seed, clen)
        v = vector[i_col] * weight_value
        while (i_row < num_row):
            out[i_row] += _dist1(seed, clen) * v
            i_row += _dist1(seed, clen)

@ti.kernel
def _mv_prob_homo_outdim_parallel_gpu(
    vector: ti.types.ndarray(ndim=1),
    weight: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    shape: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = shape[0]
    num_col = shape[1]
    weight_value = weight[0]
    for i_col in range(num_col):
        i_row = _dist1(seed, clen)
        v = vector[i_col] * weight_value
        while (i_row < num_row):
            out[i_row] += _dist1(seed, clen) * vector[i_col]
            i_row += _dist1(seed, clen)

@ti.kernel
def _mv_prob_homo_cpu(
    vector: ti.types.ndarray(ndim=1),
    weight: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    shape: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = shape[0]
    num_col = shape[1]
    weight_value = weight[0]
    ti.loop_config(serialize=True)
    for i_row in range(num_row):
        r = 0
        i_col = _dist1(seed, clen)
        while (i_col < num_col):
            r += vector[i_col]
            i_col += _dist1(seed, clen)
        out[i_row] = r * weight_value

@ti.kernel
def _mv_prob_homo_gpu(
    vector: ti.types.ndarray(ndim=1),
    weight: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    shape: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = shape[0]
    num_col = shape[1]
    weight_value = weight[0]
    for i_row in range(num_row):
        r = 0
        i_col = _dist1(seed, clen)
        while (i_col < num_col):
            r += vector[i_col]
            i_col += _dist1(seed, clen)
        out[i_row] = r * weight_value

def _mv_prob_homo_jvp(
    primals, tangents, *, shape, transpose, outdim_parallel
):
    vector, weight, clen, seed = primals
    vector_dot, weight_dot, clen_dot, seed_dot = tangents
    r = mv_prob_homo_taichi(vector,
                            weight,
                            clen,
                            seed,
                            shape=shape,
                            transpose=transpose,
                            outdim_parallel=outdim_parallel)
    
    assert type(clen_dot) is ad.Zero
    assert type(seed_dot) is ad.Zero
    if type(weight_dot) is ad.Zero:
        if type(vector_dot) is ad.Zero:
            raise ValueError
        r_dot = mv_prob_homo_taichi(vector_dot,
                                    weight,
                                    clen,
                                    seed,
                                    shape=shape,
                                    transpose=transpose,
                                    outdim_parallel=outdim_parallel)
    elif type(vector_dot) is ad.Zero:
        r_dot = mv_prob_homo_taichi(vector,
                                    weight_dot,
                                    clen,
                                    seed,
                                    shape=shape,
                                    transpose=transpose,
                                    outdim_parallel=outdim_parallel)
    else:
        r_dot = mv_prob_homo_taichi(vector_dot,
                                    weight_dot,
                                    clen,
                                    seed,
                                    shape=shape,
                                    transpose=transpose,
                                    outdim_parallel=outdim_parallel)
    
    return r, r_dot

def _mv_prob_homo_transpose(
    ct, vector, weight, clen, seed, *, shape, transpose, outdim_parallel
):
    assert type(weight) is not ad.UndefinedPrimal
    assert type(clen) is not ad.UndefinedPrimal
    assert type(seed) is not ad.UndefinedPrimal
    assert type(vector) is ad.UndefinedPrimal
    r = mv_prob_homo_taichi(ct[0],
                            weight,
                            clen,
                            seed,
                            shape=shape,
                            transpose=not transpose,
                            outdim_parallel=not outdim_parallel)[0]
    return r, weight, clen, seed

def mv_prob_homo_taichi(
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
    vector = as_jax(vector)
    weight = jnp.atleast_1d(as_jax(weight))
    conn_prob = jnp.atleast_1d(as_jax(conn_prob))
    clen = jnp.asarray(jnp.ceil(1 / conn_prob) * 2 - 1, dtype=jnp.int32)
    with jax.ensure_compile_time_eval():
        if seed is None:
            seed = int(np.random.randint(0, int(1e8)))
    seed = jnp.atleast_1d(as_jax(seed, dtype=jnp.int32))

    shape = (shape[1], shape[0]) if transpose else shape
    out_shape = shape[0]

    assert _get_dtype(vector) in [jnp.float16, jnp.float32, jnp.float64]
    assert _get_dtype(weight) in [jnp.float16, jnp.float32, jnp.float64], '"weight" must be float valued.'
    assert _get_dtype(clen) in [jnp.int16, jnp.int32, jnp.int64, jnp.uint16, jnp.uint32, jnp.uint64]
    assert _get_dtype(seed) in [jnp.int16, jnp.int32, jnp.int64, jnp.uint16, jnp.uint32, jnp.uint64]

    if vector.ndim != 1:
        raise ValueError('vector should be a 1D vector.')
    if len(shape) != 2:
        raise ValueError('shape should be a length-2 tuple.')
    if seed.ndim != 1:
        raise ValueError('seed must be a 1D scalar.')
    if clen.ndim != 1:
        raise ValueError('conn_prob must be a 1D scalar.')
    if weight.ndim != 1:
        raise ValueError('weight must be a 1D scalar.')

    if len(shape) != 2:
        raise ValueError('shape should be a length-2 tuple.')
    if not isinstance(outdim_parallel, bool):
        raise ValueError('outdim_parallel must be boolean value.')
    if not isinstance(transpose, bool):
        raise ValueError('transpose must be boolean value.')
    if transpose:
        if vector.shape[0] != shape[0]:
            raise ValueError(f'Shape mismatch, vec ({vector.shape[0]},) @ mat {shape}.')
    else:
        if vector.shape[0] != shape[1]:
            raise ValueError(f'Shape mismatch, mat {shape} @ vec ({vector.shape[0]},).')
    
    prim = None
    if outdim_parallel:
        prim = _mv_prob_homo_outdim_parallel_p
    else:
        prim = _mv_prob_homo_p
    
    return prim(vector,
                weight,
                clen,
                seed,
                outs=[jax.ShapeDtypeStruct(shape=out_shape, dtype=type(weight))],
                shape=shape,
                transpose=transpose,
                outdim_parallel=outdim_parallel)[0]

# outdim_parallel = True
_mv_prob_homo_outdim_parallel_p = XLACustomOp(cpu_kernel=_mv_prob_homo_outdim_parallel_cpu,
                                              gpu_kernel=_mv_prob_homo_outdim_parallel_gpu)
_mv_prob_homo_outdim_parallel_p.def_jvp_rule(_mv_prob_homo_jvp)
_mv_prob_homo_outdim_parallel_p.def_transpose_rule(_mv_prob_homo_transpose)

# outdim_parallel = False
_mv_prob_homo_p = XLACustomOp(cpu_kernel=_mv_prob_homo_cpu,
                              gpu_kernel=_mv_prob_homo_gpu)
_mv_prob_homo_p.def_jvp_rule(_mv_prob_homo_jvp)
_mv_prob_homo_p.def_transpose_rule(_mv_prob_homo_transpose)


@ti.func
def _dist2(seed: ti.types.ndarray(ndim=1),
           w_min: ti.types.ndarray(ndim=1),
           w_max: ti.types.ndarray(ndim=1)):
    return uniform_real_distribution(seed, w_min[0], w_max[0])

@ti.kernel
def _mv_prob_uniform_outdim_parallel_cpu(
    vector: ti.types.ndarray(ndim=1),
    w_min: ti.types.ndarray(ndim=1),
    w_max: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    shape: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = shape[0]
    num_col = shape[1]
    ti.loop_config(serialize=True)
    for i_col in range(num_col):
        i_row = _dist1(seed, clen)
        while (i_row < num_row):
            out[i_row] += _dist2(seed, w_min, w_max) * vector[i_col]
            i_row += _dist1(seed, clen)

@ti.kernel
def _mv_prob_uniform_outdim_parallel_gpu(
    vector: ti.types.ndarray(ndim=1),
    w_min: ti.types.ndarray(ndim=1),
    w_max: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    shape: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = shape[0]
    num_col = shape[1]
    for i_col in range(num_col):
        i_row = _dist1(seed, clen)
        while (i_row < num_row):
            out[i_row] += _dist2(seed, w_min, w_max) * vector[i_col]
            i_row += _dist1(seed, clen)

@ti.kernel
def _mv_prob_uniform_cpu(
    vector: ti.types.ndarray(ndim=1),
    w_min: ti.types.ndarray(ndim=1),
    w_max: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    shape: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = shape[0]
    num_col = shape[1]
    ti.loop_config(serialize=True)
    for i_row in range(num_row):
        r = 0
        i_col = _dist1(seed, clen)
        while (i_col < num_col):
            r += _dist2(seed, w_min, w_max) * vector[i_col]
            i_col += _dist1(seed, clen)
        out[i_row] = r

@ti.kernel
def _mv_prob_uniform_gpu(
    vector: ti.types.ndarray(ndim=1),
    w_min: ti.types.ndarray(ndim=1),
    w_max: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    shape: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = shape[0]
    num_col = shape[1]
    for i_row in range(num_row):
        r = 0
        i_col = _dist1(seed, clen)
        while (i_col < num_col):
            r += _dist2(seed, w_min, w_max) * vector[i_col]
            i_col += _dist1(seed, clen)
        out[i_row] = r

def _mv_prob_uniform_jvp(
    primals, tangents, *, shape, transpose, outdim_parallel, conn_prob
):
    vector, w_low, w_high, clen, seed = primals
    vector_dot, w_low_dot, w_high_dot, clen_dot, seed_dot = tangents
    r = mv_prob_uniform_taichi(vector,
                               w_low,
                               w_high,
                               conn_prob,
                               seed,
                               shape=shape,
                               transpose=transpose,
                               outdim_parallel=outdim_parallel)
    
    assert type(w_low_dot) is ad.Zero
    assert type(w_high_dot) is ad.Zero
    assert type(clen_dot) is ad.Zero
    assert type(seed_dot) is ad.Zero
    
    r_dot = mv_prob_uniform_taichi(vector_dot,
                                   w_low,
                                   w_high,
                                   conn_prob,
                                   seed,
                                   shape=shape,
                                   transpose=transpose,
                                   outdim_parallel=outdim_parallel)
    return r, r_dot

def _mv_prob_uniform_transpose(
    ct, vector, w_low, w_high, clen, seed, *, shape, transpose, outdim_parallel, conn_prob
):
    assert type(vector) is not ad.UndefinedPrimal
    assert type(w_low) is not ad.UndefinedPrimal
    assert type(w_high) is not ad.UndefinedPrimal
    assert type(clen) is not ad.UndefinedPrimal
    assert type(seed) is not ad.UndefinedPrimal
    r = mv_prob_homo_taichi(ct[0],
                            w_low,
                            w_high,
                            conn_prob,
                            seed,
                            shape=shape,
                            transpose=not transpose,
                            outdim_parallel=not outdim_parallel)[0]
    return r, w_low, w_high, clen, seed


def mv_prob_uniform_taichi(
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
    vector = as_jax(vector)
    w_low = jnp.atleast_1d(as_jax(w_low))
    w_high = jnp.atleast_1d(as_jax(w_high))
    conn_prob = jnp.atleast_1d(as_jax(conn_prob))
    clen = jnp.asarray(jnp.ceil(1 / conn_prob) * 2 - 1, dtype=jnp.int32)
    with jax.ensure_compile_time_eval():
        if seed is None:
            seed = int(np.random.randint(0, int(1e8)))
    seed = jnp.atleast_1d(as_jax(seed, dtype=jnp.int32))

    shape = (shape[1], shape[0]) if transpose else shape
    out_shape = shape[0]

    if vector.ndim != 1:
        raise ValueError('vector should be a 1D vector.')
    if len(shape) != 2:
        raise ValueError('shape should be a length-2 tuple.')
    if w_low.ndim != 1:
        raise ValueError('w_low must be a 1D scalar.')
    if w_high.ndim != 1:
        raise ValueError('w_high must be a 1D scalar.')
    if clen.ndim != 1:
        raise ValueError('clen must be a 1D scalar.')
    if seed.ndim != 1:
        raise ValueError('seed must be a 1D scalar.')

    if not isinstance(transpose, bool):
        raise ValueError('transpose must be a boolean value.')
    if not isinstance(outdim_parallel, bool):
        raise ValueError('outdim_parallel must be a boolean value.')
    assert w_low.dtype == w_high.dtype == vector.dtype

# outdim_parallel = True
_mv_prob_uniform_outdim_parallel_p = XLACustomOp(cpu_kernel=_mv_prob_uniform_outdim_parallel_cpu,
                                              gpu_kernel=_mv_prob_uniform_outdim_parallel_gpu)
_mv_prob_uniform_outdim_parallel_p.def_jvp_rule(_mv_prob_uniform_jvp)
_mv_prob_uniform_outdim_parallel_p.def_transpose_rule(_mv_prob_uniform_transpose)

# outdim_parallel = False
_mv_prob_uniform_p = XLACustomOp(cpu_kernel=_mv_prob_uniform_cpu,
                              gpu_kernel=_mv_prob_uniform_gpu)
_mv_prob_uniform_p.def_jvp_rule(_mv_prob_uniform_jvp)
_mv_prob_uniform_p.def_transpose_rule(_mv_prob_uniform_transpose)