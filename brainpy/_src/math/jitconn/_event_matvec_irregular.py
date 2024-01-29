
import jax
import jax.numpy as jnp
import numpy as np

from brainpy._src.dependency_check import import_taichi
from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import _get_dtype
from brainpy._src.math.op_register import XLACustomOp
from brainpy._src.math.tifunc import (lfsr88_key, lfsr88_uniform, lfsr88_normal, lfsr88_random_integers)
from typing import Tuple, Optional

ti = import_taichi()

__all__ = [
  'event_mv_prob_homo_irregular',
]

@ti.kernel
def _event_mv_prob_homo_bool_irregular_cpu(
        events: ti.types.ndarray(ndim=1),
        weight: ti.types.ndarray(ndim=1),
        clen: ti.types.ndarray(ndim=1),
        seed: ti.types.ndarray(ndim=1),
        out: ti.types.ndarray(ndim=2)
):
    num_row = out.shape[1]
    num_col = events.shape[0]
    weight0 = weight[0]
    clen0 = clen[0]

    for i_col in range(num_col):
        if events[i_col]:
            for i_area in range(out.shape[0]):
                key = lfsr88_key(seed[i_area] + i_col)
                key, i_row = lfsr88_random_integers(key, 0, clen0 - 1)
                while i_row < num_row:
                    out[i_area, i_row] += weight0
                    key, inc = lfsr88_random_integers(key, 1, clen0)
                    i_row += inc


@ti.kernel
def _event_mv_prob_homo_bool_irregular_gpu(
        events: ti.types.ndarray(ndim=1),
        weight: ti.types.ndarray(ndim=1),
        clen: ti.types.ndarray(ndim=1),
        seed: ti.types.ndarray(ndim=1),
        out: ti.types.ndarray(ndim=2)
):
    num_row = out.shape[1]
    num_col = events.shape[0]
    weight0 = weight[0]
    clen0 = clen[0]
    step = ti.uint32(ti.max((num_row + 1) >> 5, 1))

    for i in range(num_col * 32):
        i_col = i >> 5
        if events[i_col]:
            for i_area in range(out.shape[0]):
                index = i & 31
                i_row = step * index + 1
                end = ti.min(i_row + step, num_row)

                key = lfsr88_key(seed[i_area] + i)
                key, inc = lfsr88_random_integers(key, 0, clen0 - 1)
                i_row += inc
                while i_row < end:
                    out[i_area, i_row] += weight0
                    key, inc = lfsr88_random_integers(key, 1, clen0)
                    i_row += inc


@ti.kernel
def _event_mv_prob_homo_irregular_cpu(
        events: ti.types.ndarray(ndim=1),
        weight: ti.types.ndarray(ndim=1),
        clen: ti.types.ndarray(ndim=1),
        seed: ti.types.ndarray(ndim=1),
        out: ti.types.ndarray(ndim=2)
):
    num_row = out.shape[1]
    num_col = events.shape[0]
    weight0 = weight[0]
    clen0 = clen[0]

    for i_col in range(num_col):
        if events[i_col] != 0.:
            for i_area in range(out.shape[0]):
                key = lfsr88_key(seed[i_area] + i_col)
                key, i_row = lfsr88_random_integers(key, 0, clen0 - 1)
                while i_row < num_row:
                    out[i_area, i_row] += weight0
                    key, inc = lfsr88_random_integers(key, 1, clen0)
                    i_row += inc


@ti.kernel
def _event_mv_prob_homo_irregular_gpu(
        events: ti.types.ndarray(ndim=1),
        weight: ti.types.ndarray(ndim=1),
        clen: ti.types.ndarray(ndim=1),
        seed: ti.types.ndarray(ndim=1),
        out: ti.types.ndarray(ndim=2)
):
    num_row = out.shape[1]
    num_col = events.shape[0]
    weight0 = weight[0]
    clen0 = clen[0]
    step = ti.uint32(ti.max((num_row + 1) >> 5, 1))

    for i in range(num_col * 32):
        i_col = i >> 5
        if events[i_col] != 0.:
            for i_area in range(out.shape[0]):
                index = i & 31
                i_row = step * index + 1
                end = ti.min(i_row + step, num_row)

                key = lfsr88_key(seed[i_area] + i)
                key, inc = lfsr88_random_integers(key, 0, clen0 - 1)
                i_row += inc
                while i_row < end:
                    out[i_area, i_row] += weight0
                    key, inc = lfsr88_random_integers(key, 1, clen0)
                    i_row += inc


def _reverse(shape):
    return shape[::-1]


def _general_checking(vector, clen, seed, shape, area_num, transpose, outdim_parallel, *weights):
    if vector.ndim != 1:
        raise ValueError('vector should be a 1D vector.')
    if len(shape) != 2:
        raise ValueError('shape should be a length-2 tuple.')
    if seed.ndim != 1:
        raise ValueError('seed must be a 1D scalar.')
    if clen.ndim != 1:
        raise ValueError('conn_prob must be a 1D scalar.')
    if type(area_num) != int:
        raise ValueError('area_num must be an integer.')

    assert _get_dtype(clen) in [jnp.int16, jnp.int32, jnp.int64, jnp.uint16, jnp.uint32, jnp.uint64]
    assert _get_dtype(seed) in [jnp.int16, jnp.int32, jnp.int64, jnp.uint16, jnp.uint32, jnp.uint64]

    for weight in weights:
        if weight.ndim != 1:
            raise ValueError('weight must be a 1D scalar.')
        assert _get_dtype(weight) in [jnp.float16, jnp.float32, jnp.float64], '"weight" must be float valued.'

    if not isinstance(outdim_parallel, bool):
        raise ValueError('outdim_parallel must be boolean value.')
    if not isinstance(transpose, bool):
        raise ValueError('transpose must be boolean value.')

    if transpose:
        out_shape = (area_num, shape[1])
        if vector.shape[0] != shape[0]:
            raise ValueError(f'Shape mismatch, vec {vector.shape} @ mat {shape}.')
        shape = _reverse(shape)
    else:
        if vector.shape[0] != shape[1]:
            raise ValueError(f'Shape mismatch, mat {shape} @ vec ({vector.shape[0]},).')
        out_shape = (area_num, shape[0])

    return shape, out_shape


def _event_checking(vector, clen, seed, shape, area_num, transpose, outdim_parallel, *weights):
    assert _get_dtype(vector) in [jnp.bool_, jnp.float16, jnp.float32, jnp.float64]
    return _general_checking(vector, clen, seed, shape, area_num, transpose, outdim_parallel, *weights)


def _define_event_mv_prob_homo_irregular_prim(cpu_kernel, gpu_kernel):
    prim = XLACustomOp(cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel)
    return prim


_event_mv_prob_homo_bool_irregular_p = _define_event_mv_prob_homo_irregular_prim(
    cpu_kernel=_event_mv_prob_homo_bool_irregular_cpu,
    gpu_kernel=_event_mv_prob_homo_bool_irregular_gpu
)

_event_mv_prob_homo_irregular_p = _define_event_mv_prob_homo_irregular_prim(
    cpu_kernel=_event_mv_prob_homo_irregular_cpu,
    gpu_kernel=_event_mv_prob_homo_irregular_gpu
)


def raw_event_mv_prob_homo_irregular(
        events: jax.Array,
        weight: jax.Array,  # vector with size 1
        conn_len: jax.Array,  # vector with size 1
        seed: jax.Array,  # vector with size 1
        *,
        shape: Tuple[int, int],
        area_num: int,
        transpose: bool = False,
        outdim_parallel: bool = False,
) -> jax.Array:
    mat_shape, out_shape = _event_checking(events, conn_len, seed, shape, area_num, transpose, outdim_parallel, weight)

    if outdim_parallel:
        raise NotImplementedError('Not implement outdim_parallel=True')
    else:
        if events.dtype == jnp.bool_:
            prim = _event_mv_prob_homo_bool_irregular_p
        else:
            prim = _event_mv_prob_homo_irregular_p
    
    return prim(events,
                weight,
                conn_len,
                seed,
                outs=[jax.ShapeDtypeStruct(shape=out_shape, dtype=weight.dtype)],
                shape=mat_shape,
                area_num=area_num,
                transpose=transpose,
                outdim_parallel=outdim_parallel)

def event_mv_prob_homo_irregular(
        events: jax.Array,
        weight: float,
        conn_prob: float,
        seed: Optional[int] = None,
        *,
        shape: Tuple[int, int],
        area_num: int,
        transpose: bool = False,
        outdim_parallel: bool = False,
):
    events = as_jax(events)
    if isinstance(weight, float): weight = as_jax(weight)
    weight = jnp.atleast_1d(as_jax(weight))
    conn_len = jnp.ceil(1 / conn_prob) * 2 - 1
    conn_len = jnp.asarray(jnp.atleast_1d(conn_len), dtype=jnp.int32)
    if seed is None:
        with jax.ensure_compile_time_eval():
            seed = np.random.randint(0, int(1e8), 1)
    seed = jnp.atleast_1d(jnp.asarray(seed, dtype=jnp.uint32))
    return raw_event_mv_prob_homo_irregular(events, weight, conn_len, seed, shape=shape, area_num=area_num,
                                            transpose=transpose, outdim_parallel=outdim_parallel)[0]