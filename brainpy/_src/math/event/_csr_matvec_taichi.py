# -*- coding: utf-8 -*-

from functools import partial
from typing import Union, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import taichi as ti
from jax.core import ShapedArray, Primitive
from jax.interpreters import ad, xla
from jax.lib import xla_client

from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.op_register import (XLACustomOp,
                                      register_general_batching)
from brainpy._src.math.sparse._csr_mv import csrmv as normal_csrmv
from brainpy._src.math.sparse._utils import csr_to_coo
from brainpy._src.dependency_check import (import_brainpylib_cpu_ops,
                                           import_brainpylib_gpu_ops)
from brainpy.errors import GPUOperatorNotFound

__all__ = [
    'csrmv_taichi'
]

@ti.kernel
def event_csr_matvec_cpu_transpose(values: ti.types.ndarray(ndim=1),
                         indices: ti.types.ndarray(ndim=1),
                         indptr: ti.types.ndarray(ndim=1),
                         events: ti.types.ndarray(ndim=1),
                         is_event_type_bool: ti.types.ndarray(ndim=1),
                         is_heter: ti.types.ndarray(ndim=1),
                         out: ti.types.ndarray(ndim=1)):
  is_event_type_bool_value = is_event_type_bool[None]
  is_heter_value = is_heter[None]
  if is_event_type_bool_value: # type of events is boolean
    if is_heter_value: # heter
      ti.loop_config(serialize=True)
      for row_i in range(events.shape[0]):
        if events[row_i]:
          for j in range(indptr[row_i], indptr[row_i + 1]):
            out[indices[j]] += values[j]
      

    else: # homo
      value = values[0]
      ti.loop_config(serialize=True)
      for row_i in range(events.shape[0]):
        if events[row_i]:
          for j in range(indptr[row_i], indptr[row_i + 1]):
            out[indices[j]] += value

  else: # type of events is not boolean
    if is_heter_value: # heter
      ti.loop_config(serialize=True)
      for row_i in range(events.shape[0]):
        if events[row_i] > 0.:
          for j in range(indptr[row_i], indptr[row_i + 1]):
            out[indices[j]] += values[j]

    else: # homo
      value = values[0]
      ti.loop_config(serialize=True)
      for row_i in range(events.shape[0]):
        if events[row_i] > 0.:
          for j in range(indptr[row_i], indptr[row_i + 1]):
            out[indices[j]] += value


@ti.kernel
def event_csr_matvec_cpu(values: ti.types.ndarray(ndim=1),
                         indices: ti.types.ndarray(ndim=1),
                         indptr: ti.types.ndarray(ndim=1),
                         events: ti.types.ndarray(ndim=1),
                         is_event_type_bool: ti.types.ndarray(ndim=1),
                         is_heter: ti.types.ndarray(ndim=1),
                         out: ti.types.ndarray(ndim=1)):
  is_event_type_bool_value = is_event_type_bool[None]
  is_heter_value = is_heter[None]
  if is_event_type_bool_value: # type of events is boolean
    if is_heter_value: # heter
      ti.loop_config(serialize=True)
      for row_i in range(out.shape[0]):
        r = 0.
        for j in range(indptr[row_i], indptr[row_i + 1]):
          if events[indices[j]]:
            r += values[j]
        out[row_i] = r

    else: # homo
      value = values[0]
      ti.loop_config(serialize=True)
      for row_i in range(out.shape[0]):
        r = 0.
        for j in range(indptr[row_i], indptr[row_i + 1]):
          if events[indices[j]]:
            r += value
        out[row_i] = r
      

  else: # type of events is not boolean
    if is_heter_value: # heter
      ti.loop_config(serialize=True)
      for row_i in range(out.shape[0]):
        r = 0.
        for j in range(indptr[row_i], indptr[row_i + 1]):
          if events[indices[j]] > 0.:
            r += values[j]
        out[row_i] = r

    else: # homo
      value = values[0]
      ti.loop_config(serialize=True)
      for row_i in range(out.shape[0]):
        r = 0.
        for j in range(indptr[row_i], indptr[row_i + 1]):
          if events[indices[j]] > 0.:
            r += value
        out[row_i] = r

@ti.kernel
def event_csr_matvec_gpu_transpose(values: ti.types.ndarray(),
                         indices: ti.types.ndarray(),
                         indptr: ti.types.ndarray(),
                         events: ti.types.ndarray(),
                         is_event_type_bool: ti.types.ndarray(),
                         is_heter: ti.types.ndarray(),
                         out: ti.types.ndarray()):
  is_event_type_bool_value = is_event_type_bool[None]
  is_heter_value = is_heter[None]
  if is_event_type_bool[0]: # type of events is boolean
    if is_heter[0]: # heter
      for row_i in range(events):
        if events[row_i]:
          for j in range(indptr[row_i], indptr[row_i + 1]):
            out[indices[j]] += values[j]
      

    else: # homo
      value = values[0]
      for row_i in range(events):
        if events[row_i]:
          for j in range(indptr[row_i], indptr[row_i + 1]):
            out[indices[j]] += value

  else: # type of events is not boolean
    if is_heter[0]: # heter
      for row_i in range(events):
        if events[row_i] > 0.:
          for j in range(indptr[row_i], indptr[row_i + 1]):
            out[indices[j]] += values[j]

    else: # homo
      value = values[0]
      for row_i in range(events):
        if events[row_i] > 0.:
          for j in range(indptr[row_i], indptr[row_i + 1]):
            out[indices[j]] += value


@ti.kernel
def event_csr_matvec_gpu(values: ti.types.ndarray(),
                         indices: ti.types.ndarray(),
                         indptr: ti.types.ndarray(),
                         events: ti.types.ndarray(),
                         is_event_type_bool: ti.types.ndarray(),
                         is_heter: ti.types.ndarray(),
                         out: ti.types.ndarray()):
  is_event_type_bool_value = is_event_type_bool[None]
  is_heter_value = is_heter[None]
  if is_event_type_bool[0]: # type of events is boolean
    if is_heter[0]: # heter
      for row_i in range(out.shape[0]):
        r = 0.
        for j in range(indptr[row_i], indptr[row_i + 1]):
          if events[indices[j]]:
            r += values[j]
        out[row_i] = r

    else: # homo
      value = values[0]
      for row_i in range(out.shape[0]):
        r = 0.
        for j in range(indptr[row_i], indptr[row_i + 1]):
          if events[indices[j]]:
            r += value
      out[row_i] = r
      

  else: # type of events is not boolean
    if is_heter[0]: # heter
      for row_i in range(out.shape[0]):
        r = 0.
        for j in range(indptr[row_i], indptr[row_i + 1]):
          if events[indices[j]] > 0.:
            r += values[j]
        out[row_i] = r

    else: # homo
      value = values[0]
      for row_i in range(out.shape[0]):
        r = 0.
        for j in range(indptr[row_i], indptr[row_i + 1]):
          if events[indices[j]] > 0.:
            r += value
      out[row_i] = r

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
    raise ValueError('indices should be a 1D vector with int8, int16, int32, int64, uint8, uint16, uint32 or uint64 type.')
  if indptr.dtype not in [jnp.int8, jnp.int16, jnp.int32, jnp.int64, jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64]:
    raise ValueError('indptr should be a 1D vector with int8, int16, int32, int64, uint8, uint16, uint32 or uint64 type.')
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
    
  is_event_type_bool = jnp.array(events.dtype == jnp.bool_)
  is_heter = jnp.array(data.shape[0] > 1)

  event_csr_matvec_p = None
  if transpose:
    event_csr_matvec_p = XLACustomOp(cpu_kernel=event_csr_matvec_cpu_transpose, gpu_kernel=event_csr_matvec_gpu_transpose)
  else:
    event_csr_matvec_p = XLACustomOp(cpu_kernel=event_csr_matvec_cpu, gpu_kernel=event_csr_matvec_gpu)

  # computing
  return event_csr_matvec_p(data, indices, indptr, events, is_event_type_bool, is_heter, outs=[jax.ShapeDtypeStruct(shape=(shape[1] if transpose else shape[0],), dtype=data.dtype)])
