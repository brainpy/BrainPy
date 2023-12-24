# -*- coding: utf-8 -*-

from typing import Union, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.interpreters import ad

from brainpy._src.dependency_check import import_taichi
from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.op_register import XLACustomOp
from brainpy._src.math.sparse._csr_mv_taichi import csrmv_taichi as normal_csrmv_taichi
from brainpy._src.math.sparse._utils import csr_to_coo

ti = import_taichi()

__all__ = [
  'csrmv_taichi'
]


# -------------
# CPU operators
# -------------

# 1. The benchmarking shows that the performance of the following transpose
#    kernels is maximized when using serialized mode
# 2. Since our Taichi-JAX kernel does not support the non-differentiable/non-jittable
#    arguments, we have to define each kernel separately when the
#    non-differentiable/non-jittable arguments are different.


@ti.kernel
def _event_csr_matvec_transpose_bool_homo_cpu(values: ti.types.ndarray(ndim=1),
                                              indices: ti.types.ndarray(ndim=1),
                                              indptr: ti.types.ndarray(ndim=1),
                                              events: ti.types.ndarray(ndim=1),
                                              out: ti.types.ndarray(ndim=1)):
  value = values[0]
  ti.loop_config(serialize=True)
  for row_i in range(indptr.shape[0] - 1):
    if events[row_i]:
      for j in range(indptr[row_i], indptr[row_i + 1]):
        out[indices[j]] += value


@ti.kernel
def _event_csr_matvec_transpose_bool_heter_cpu(values: ti.types.ndarray(ndim=1),
                                               indices: ti.types.ndarray(ndim=1),
                                               indptr: ti.types.ndarray(ndim=1),
                                               events: ti.types.ndarray(ndim=1),
                                               out: ti.types.ndarray(ndim=1)):
  ti.loop_config(serialize=True)
  for row_i in range(indptr.shape[0] - 1):
    if events[row_i]:
      for j in range(indptr[row_i], indptr[row_i + 1]):
        out[indices[j]] += values[j]


@ti.kernel
def _event_csr_matvec_transpose_homo_cpu(values: ti.types.ndarray(ndim=1),
                                         indices: ti.types.ndarray(ndim=1),
                                         indptr: ti.types.ndarray(ndim=1),
                                         events: ti.types.ndarray(ndim=1),
                                         out: ti.types.ndarray(ndim=1)):
  value = values[0]
  ti.loop_config(serialize=True)
  for row_i in range(indptr.shape[0] - 1):
    if events[row_i] != 0.:
      for j in range(indptr[row_i], indptr[row_i + 1]):
        out[indices[j]] += value


@ti.kernel
def _event_csr_matvec_transpose_heter_cpu(values: ti.types.ndarray(ndim=1),
                                          indices: ti.types.ndarray(ndim=1),
                                          indptr: ti.types.ndarray(ndim=1),
                                          events: ti.types.ndarray(ndim=1),
                                          out: ti.types.ndarray(ndim=1)):
  ti.loop_config(serialize=True)
  for row_i in range(indptr.shape[0] - 1):
    if events[row_i] != 0.:
      for j in range(indptr[row_i], indptr[row_i + 1]):
        out[indices[j]] += values[j]


@ti.kernel
def _event_csr_matvec_bool_homo_cpu(values: ti.types.ndarray(ndim=1),
                                    indices: ti.types.ndarray(ndim=1),
                                    indptr: ti.types.ndarray(ndim=1),
                                    events: ti.types.ndarray(ndim=1),
                                    out: ti.types.ndarray(ndim=1)):
  value = values[0]
  # ti.loop_config(serialize=True)
  for row_i in range(indptr.shape[0] - 1):
    r = 0.
    for j in range(indptr[row_i], indptr[row_i + 1]):
      if events[indices[j]]:
        r += value
    out[row_i] = r


@ti.kernel
def _event_csr_matvec_bool_heter_cpu(values: ti.types.ndarray(ndim=1),
                                     indices: ti.types.ndarray(ndim=1),
                                     indptr: ti.types.ndarray(ndim=1),
                                     events: ti.types.ndarray(ndim=1),
                                     out: ti.types.ndarray(ndim=1)):
  # ti.loop_config(serialize=True)
  for row_i in range(indptr.shape[0] - 1):
    r = 0.
    for j in range(indptr[row_i], indptr[row_i + 1]):
      if events[indices[j]]:
        r += values[j]
    out[row_i] = r


@ti.kernel
def _event_csr_matvec_homo_cpu(values: ti.types.ndarray(ndim=1),
                               indices: ti.types.ndarray(ndim=1),
                               indptr: ti.types.ndarray(ndim=1),
                               events: ti.types.ndarray(ndim=1),
                               out: ti.types.ndarray(ndim=1)):
  value = values[0]
  # ti.loop_config(serialize=True)
  for row_i in range(indptr.shape[0] - 1):
    r = 0.
    for j in range(indptr[row_i], indptr[row_i + 1]):
      if events[indices[j]] != 0.:
        r += value
    out[row_i] = r


@ti.kernel
def _event_csr_matvec_heter_cpu(values: ti.types.ndarray(ndim=1),
                                indices: ti.types.ndarray(ndim=1),
                                indptr: ti.types.ndarray(ndim=1),
                                events: ti.types.ndarray(ndim=1),
                                out: ti.types.ndarray(ndim=1)):
  # ti.loop_config(serialize=True)
  for row_i in range(indptr.shape[0] - 1):
    r = 0.
    for j in range(indptr[row_i], indptr[row_i + 1]):
      if events[indices[j]] != 0.:
        r += values[j]
    out[row_i] = r


# -------------
# GPU operators
# -------------

# 1. GPU kernels are different from the CPU ones, since the GPU kernels need
#    to use warp-level parallelism to achieve the best performance.


@ti.kernel
def _event_csr_matvec_transpose_bool_homo_gpu(values: ti.types.ndarray(ndim=1),
                                              indices: ti.types.ndarray(ndim=1),
                                              indptr: ti.types.ndarray(ndim=1),
                                              events: ti.types.ndarray(ndim=1),
                                              out: ti.types.ndarray(ndim=1)):
  value = values[0]
  for i in range((indptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    if events[row_i]:
      j = indptr[row_i] + index
      end_index = indptr[row_i + 1]
      while j < end_index:
        out[indices[j]] += value
        j += 32


@ti.kernel
def _event_csr_matvec_transpose_homo_gpu(values: ti.types.ndarray(ndim=1),
                                         indices: ti.types.ndarray(ndim=1),
                                         indptr: ti.types.ndarray(ndim=1),
                                         events: ti.types.ndarray(ndim=1),
                                         out: ti.types.ndarray(ndim=1)):
  value = values[0]
  for i in range((indptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    if events[row_i] != 0.:
      j = indptr[row_i] + index
      end_index = indptr[row_i + 1]
      while j < end_index:
        out[indices[j]] += value
        j += 32


# TODO
# It is important to note that the following warp-based kernels
# should be improved, since the atomic_add for each thread is not
# very efficient. Instead, the warp-level reduction primitive
# should be used.
# see ``warp_reduce_sum()`` function in tifunc.py.
# However, currently Taichi does not support general warp-level primitives.


@ti.kernel
def _event_csr_matvec_bool_homo_gpu(values: ti.types.ndarray(ndim=1),
                                    indices: ti.types.ndarray(ndim=1),
                                    indptr: ti.types.ndarray(ndim=1),
                                    events: ti.types.ndarray(ndim=1),
                                    out: ti.types.ndarray(ndim=1)):
  value = values[0]
  for i in range((indptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    r = 0.
    j = indptr[row_i] + index
    end_index = indptr[row_i + 1]
    while j < end_index:
      if events[indices[j]]:
        r += value
      j += 32
    out[row_i] += r  # TODO: warp-level primitive


@ti.kernel
def _event_csr_matvec_homo_gpu(values: ti.types.ndarray(ndim=1),
                               indices: ti.types.ndarray(ndim=1),
                               indptr: ti.types.ndarray(ndim=1),
                               events: ti.types.ndarray(ndim=1),
                               out: ti.types.ndarray(ndim=1)):
  value = values[0]
  for i in range((indptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    r = 0.
    j = indptr[row_i] + index
    end_index = indptr[row_i + 1]
    while j < end_index:
      if events[indices[j]] != 0.:
        r += value
      j += 32
    out[row_i] += r  # TODO: warp-level primitive


@ti.kernel
def _event_csr_matvec_transpose_bool_heter_gpu(values: ti.types.ndarray(ndim=1),
                                               indices: ti.types.ndarray(ndim=1),
                                               indptr: ti.types.ndarray(ndim=1),
                                               events: ti.types.ndarray(ndim=1),
                                               out: ti.types.ndarray(ndim=1)):
  for i in range((indptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    if events[row_i]:
      j = indptr[row_i] + index
      end_index = indptr[row_i + 1]
      while j < end_index:
        out[indices[j]] += values[j]
        j += 32


@ti.kernel
def _event_csr_matvec_transpose_heter_gpu(values: ti.types.ndarray(ndim=1),
                                          indices: ti.types.ndarray(ndim=1),
                                          indptr: ti.types.ndarray(ndim=1),
                                          events: ti.types.ndarray(ndim=1),
                                          out: ti.types.ndarray(ndim=1)):
  for i in range((indptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    if events[row_i] != 0.:
      j = indptr[row_i] + index
      end_index = indptr[row_i + 1]
      while j < end_index:
        out[indices[j]] += values[j]
        j += 32


@ti.kernel
def _event_csr_matvec_bool_heter_gpu(values: ti.types.ndarray(ndim=1),
                                     indices: ti.types.ndarray(ndim=1),
                                     indptr: ti.types.ndarray(ndim=1),
                                     events: ti.types.ndarray(ndim=1),
                                     out: ti.types.ndarray(ndim=1)):
  for i in range((indptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    r = 0.
    j = indptr[row_i] + index
    end_index = indptr[row_i + 1]
    while j < end_index:
      if events[indices[j]]:
        r += values[j]
      j += 32
    out[row_i] += r   # TODO: warp-level primitive


@ti.kernel
def _event_csr_matvec_heter_gpu(values: ti.types.ndarray(ndim=1),
                                indices: ti.types.ndarray(ndim=1),
                                indptr: ti.types.ndarray(ndim=1),
                                events: ti.types.ndarray(ndim=1),
                                out: ti.types.ndarray(ndim=1)):
  for i in range((indptr.shape[0] - 1) * 32):
    row_i = i >> 5
    index = i & 31
    r = 0.
    j = indptr[row_i] + index
    end_index = indptr[row_i + 1]
    while j < end_index:
      if events[indices[j]] != 0.:
        r += values[j]
      j += 32
    out[row_i] += r   # TODO: warp-level primitive


def _event_csr_matvec_jvp_values(val_dot, values, indices, indptr, events, *, outs, transpose, shape):
  return normal_csrmv_taichi(val_dot, indices, indptr, events, shape=shape, transpose=transpose)


def _event_csr_matvec_jvp_events(evt_dot, values, indices, indptr, events, *, outs, transpose, shape):
  return normal_csrmv_taichi(values, indices, indptr, evt_dot, shape=shape, transpose=transpose)


def _event_csr_matvec_transpose(
    ct, values, indices, indptr, events, *, outs, transpose, shape
):
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices.")
  if ad.is_undefined_primal(events):
    ct_events = normal_csrmv_taichi(values, indices, indptr, ct[0], shape=shape, transpose=transpose)[0]
    return values, indices, indptr, (ad.Zero(events) if type(ct[0]) is ad.Zero else ct_events)
  else:
    if type(ct[0]) is ad.Zero:
      ct_values = ad.Zero(values)
    else:
      if values.aval.shape[0] == 1:  # scalar
        ct_values = csrmv_taichi(jnp.ones(1), indices, indptr, events, shape=shape, transpose=transpose)[0]
        ct_values = jnp.inner(ct[0], ct_values)
      else:  # heterogeneous values
        row, col = csr_to_coo(indices, indptr)
        ct_values = events[row] * ct[0][col] if transpose else events[col] * ct[0][row]
    return ct_values, indices, indptr, events


def csrmv_taichi(
    data: Union[float, jax.Array],
    indices: jax.Array,
    indptr: jax.Array,
    events: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False
) -> jax.Array:
  """Product of a sparse CSR matrix and a dense event vector.

  This function supports JAX transformations, including `jit()`, `grad()`,
  `vmap()` and `pmap()`.

  Parameters
  ----------
  data: ndarray, float
    An array of shape ``(nse,)``.
  indices: ndarray
    An array of shape ``(nse,)``.
  indptr: ndarray
    An array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``.
  events: ndarray
    An array of shape ``(shape[0] if transpose else shape[1],)``
    and dtype ``data.dtype``.
  shape: tuple
    A length-2 tuple representing the matrix shape.
  transpose: bool
    A boolean specifying whether to transpose the sparse matrix
    before computing.
    If ``transpose=True``, the operator will compute based on the
    event-driven property of the ``events`` vector.

  Returns
  -------
  y : Array
    The array of shape ``(shape[1] if transpose else shape[0],)`` representing
    the matrix vector product.
  """
  data = as_jax(data)
  indices = as_jax(indices)
  indptr = as_jax(indptr)
  events = as_jax(events)

  # checking
  data = jnp.atleast_1d(data)
  if np.ndim(data) == 1:
    if data.shape[0] not in [1, indices.shape[0]]:
      raise ValueError('The size of data should be 1 or be consistent with indices.'
                       f'But we got {data.shape} != {indices.shape}, {data.shape} != 1.')
  else:
    raise ValueError('data should be a scalar or 1D vector. '
                     f'But we got {np.ndim(data)}-D array.')
  if np.ndim(indices) != 1:
    raise ValueError('indices should be a 1D vector with integer type.')
  if np.ndim(indptr) != 1:
    raise ValueError('indptr should be a 1D vector with integer type.')
  if indices.dtype not in [jnp.int8, jnp.int16, jnp.int32, jnp.int64, jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64]:
    raise ValueError(
      'indices should be a 1D vector with int8, int16, int32, int64, uint8, uint16, uint32 or uint64 type.')
  if indptr.dtype not in [jnp.int8, jnp.int16, jnp.int32, jnp.int64, jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64]:
    raise ValueError(
      'indptr should be a 1D vector with int8, int16, int32, int64, uint8, uint16, uint32 or uint64 type.')
  if np.ndim(events) != 1:
    raise ValueError('events should be a 1D vector.')
  if len(shape) != 2:
    raise ValueError('shape should be a length-2 tuple.')
  if transpose:
    if events.shape[0] != shape[0]:
      raise ValueError(f'Shape mismatch, vec ({events.shape[0]},) @ mat {shape}.')
  else:
    if events.shape[0] != shape[1]:
      raise ValueError(f'Shape mismatch, mat {shape} @ vec ({events.shape[0]},).')

  # if the shape of indices is (0,), then we return a zero vector
  if indices.shape[0] == 0:
    return jnp.zeros(shape[1] if transpose else shape[0], dtype=data.dtype)

  if transpose:
    if events.dtype == jnp.bool_:
      if data.shape[0] == 1:
        prim = _event_csrmv_transpose_bool_homo_p
      else:
        prim = _event_csrmv_transpose_bool_heter_p
    else:
      if data.shape[0] == 1:
        prim = _event_csrmv_transpose_homo_p
      else:
        prim = _event_csrmv_transpose_heter_p
  else:
    if events.dtype == jnp.bool_:
      if data.shape[0] == 1:
        prim = _event_csrmv_bool_homo_p
      else:
        prim = _event_csrmv_bool_heter_p
    else:
      if data.shape[0] == 1:
        prim = _event_csrmv_homo_p
      else:
        prim = _event_csrmv_heter_p

  # computing
  return prim(data,
              indices,
              indptr,
              events,
              outs=[jax.ShapeDtypeStruct(shape=(shape[1] if transpose else shape[0],), dtype=data.dtype)],
              transpose=transpose,
              shape=shape)


def _define_op(cpu_kernel, gpu_kernel):
  prim = XLACustomOp(cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel)
  prim.defjvp(_event_csr_matvec_jvp_values, None, None, _event_csr_matvec_jvp_events)
  prim.def_transpose_rule(_event_csr_matvec_transpose)
  return prim


# transpose bool homo
_event_csrmv_transpose_bool_homo_p = _define_op(_event_csr_matvec_transpose_bool_homo_cpu,
                                                _event_csr_matvec_transpose_bool_homo_gpu)

# transpose homo
_event_csrmv_transpose_homo_p = _define_op(_event_csr_matvec_transpose_homo_cpu, _event_csr_matvec_transpose_homo_gpu)

# not transpose bool homo
_event_csrmv_bool_homo_p = _define_op(_event_csr_matvec_bool_homo_cpu, _event_csr_matvec_bool_homo_gpu)

# not transpose homo
_event_csrmv_homo_p = _define_op(_event_csr_matvec_homo_cpu, _event_csr_matvec_homo_gpu)

# transpose bool heter
_event_csrmv_transpose_bool_heter_p = _define_op(_event_csr_matvec_transpose_bool_heter_cpu,
                                                 _event_csr_matvec_transpose_bool_heter_gpu)

# transpose heter
_event_csrmv_transpose_heter_p = _define_op(_event_csr_matvec_transpose_heter_cpu,
                                            _event_csr_matvec_transpose_heter_gpu)

# not transpose bool heter
_event_csrmv_bool_heter_p = _define_op(_event_csr_matvec_bool_heter_cpu, _event_csr_matvec_bool_heter_gpu)

# not transpose heter
_event_csrmv_heter_p = _define_op(_event_csr_matvec_heter_cpu, _event_csr_matvec_heter_gpu)
