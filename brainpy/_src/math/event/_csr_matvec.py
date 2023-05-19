# -*- coding: utf-8 -*-


from functools import partial
from typing import Union, Tuple

import jax
import jax.numpy as jnp
import numba
import numpy as np
from jax.core import ShapedArray, Primitive
from jax.interpreters import ad, xla
from jax.lib import xla_client

from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.op_registers import (compile_cpu_signature_with_numba,
                                            register_general_batching)
from brainpy._src.math.sparse._csr_mv import csrmv as normal_csrmv
from brainpy._src.math.sparse._utils import csr_to_coo
from brainpy.errors import GPUOperatorNotFound

try:
  from brainpylib import gpu_ops
except ImportError:
  gpu_ops = None

__all__ = [
  'csrmv'
]


def csrmv(
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

  Returns
  -------
  y : ndarry
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
  if indices.dtype not in [jnp.int32, jnp.uint32]:
    raise ValueError('indices should be a 1D vector with int32 or uint32 type.')
  if indptr.dtype not in [jnp.int32, jnp.uint32]:
    raise ValueError('indptr should be a 1D vector with int32 or uint32 type.')
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

  # computing
  return event_csr_matvec_p.bind(data, indices, indptr, events, shape=shape, transpose=transpose)


# ----------------------------------------------------------
# event csr matvec
# ----------------------------------------------------------

# operator for `event_csr_matvec` batching rule
# --------

def _batch_event_csr_matvec_abstract(
    values, indices, indptr, events, *, batch_size, shape, transpose=False
):
  return ShapedArray(dtype=values.dtype, shape=(batch_size, shape[1] if transpose else shape[0]))


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _batch_event_csr_matvec_transpose_numba_imp(outs, ins):
  res_val = outs
  res_val.fill(0)
  values, indices, indptr, events, batch_size, shape, _ = ins
  batch_size = batch_size[()]
  event_batch_dim = events.shape[0]
  indices_batch_dim = indices.shape[0]
  indptr_batch_dim = indptr.shape[0]
  values_batch_dim = values.shape[0]

  if values.shape[1] == 1:  # homogeneous value
    for bi in numba.prange(batch_size):
      event_bi = bi % event_batch_dim
      indptr_bi = bi % indptr_batch_dim
      indices_bi = bi % indices_batch_dim
      values_bi = bi % values_batch_dim
      for row_i in range(shape[0]):
        if events[event_bi, row_i]:
          value = values[values_bi, 0]
          for j in range(indptr[indptr_bi, row_i], indptr[indptr_bi, row_i + 1]):
            col_i = indices[indices_bi, j]
            res_val[bi, col_i] += value

  else:  # heterogeneous values
    for bi in numba.prange(batch_size):
      event_bi = bi % event_batch_dim
      indptr_bi = bi % indptr_batch_dim
      indices_bi = bi % indices_batch_dim
      value_bi = bi % values_batch_dim
      for row_i in range(shape[0]):
        if events[event_bi, row_i]:
          for j in range(indptr[indptr_bi, row_i], indptr[indptr_bi, row_i + 1]):
            col_i = indices[indices_bi, j]
            res_val[bi, col_i] += values[value_bi, j]


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _batch_event_csr_matvec_numba_imp(outs, ins):
  res_val = outs
  res_val.fill(0)
  values, indices, indptr, events, batch_size, shape, transpose = ins
  batch_size = batch_size[()]
  event_batch_dim = events.shape[0]
  indices_batch_dim = indices.shape[0]
  indptr_batch_dim = indptr.shape[0]
  values_batch_dim = values.shape[0]

  if values.shape[1] == 1:  # homogeneous value
    for bi in numba.prange(batch_size):
      event_bi = bi % event_batch_dim
      indptr_bi = bi % indptr_batch_dim
      indices_bi = bi % indices_batch_dim
      value_bi = bi % values_batch_dim
      value = values[value_bi, 0]
      for row_i in numba.prange(shape[0]):
        r = 0.
        for j in range(indptr[indptr_bi, row_i], indptr[indptr_bi, row_i + 1]):
          col_i = indices[indices_bi, j]
          if events[event_bi, col_i]:
            r += value
        res_val[bi, row_i] = r

  else:  # heterogeneous values
    for bi in numba.prange(batch_size):
      event_bi = bi % event_batch_dim
      indptr_bi = bi % indptr_batch_dim
      indices_bi = bi % indices_batch_dim
      value_bi = bi % values_batch_dim
      for row_i in numba.prange(shape[0]):
        r = 0.
        for j in range(indptr[indptr_bi, row_i], indptr[indptr_bi, row_i + 1]):
          col_i = indices[indices_bi, j]
          if events[event_bi, col_i]:
            r += values[value_bi, j]
        res_val[bi, row_i] = r


def _batch_event_csr_matvec_cpu_translation(c, values, indices, indptr, events, *,
                                            batch_size, shape, transpose):
  inputs = (values, indices, indptr, events)
  description = dict(batch_size=batch_size, shape=shape, transpose=transpose)
  if transpose:
    name, inputs, in_layouts, out_layouts = compile_cpu_signature_with_numba(
      c,
      _batch_event_csr_matvec_transpose_numba_imp,
      _batch_event_csr_matvec_abstract,
      False,
      inputs=inputs,
      description=description
    )
  else:
    name, inputs, in_layouts, out_layouts = compile_cpu_signature_with_numba(
      c,
      _batch_event_csr_matvec_numba_imp,
      _batch_event_csr_matvec_abstract,
      False,
      inputs=inputs,
      description=description
    )
  return xla_client.ops.CustomCallWithLayout(
    c,
    name,
    operands=inputs,
    operand_shapes_with_layout=in_layouts,
    shape_with_layout=out_layouts,
  )


def _batch_event_csr_matvec_gpu_translation(c, values, indices, indptr, events, *,
                                            batch_size, shape, transpose):
  pass


def _batch_event_csr_matvec_jvp_values(values_dot, values, indices, indptr, events, *,
                                       batch_size, shape, transpose):
  return event_csr_matvec_batching_p.bind(values_dot, indices, indptr, events,
                                          batch_size=batch_size, shape=shape, transpose=transpose)


def _batch_csr_matvec(values, indices, indptr, vectors, *, shape, transpose):
  f = jax.vmap(partial(normal_csrmv, shape=shape, transpose=transpose),
               in_axes=(0 if values.shape[0] > 1 else None,
                        0 if indices.shape[0] > 1 else None,
                        0 if indptr.shape[0] > 1 else None,
                        0 if vectors.shape[0] > 1 else None))
  return f(values if values.shape[0] > 1 else values[0],
           indices if indices.shape[0] > 1 else indices[0],
           indptr if indptr.shape[0] > 1 else indptr[0],
           vectors if vectors.shape[0] > 1 else vectors[0])


def _batch_event_csr_matvec_jvp_events(events_dot, values, indices, indptr, events, *,
                                       batch_size, shape, transpose):
  return _batch_csr_matvec(values, indices, indptr, events_dot,
                           shape=shape, transpose=transpose)


def _batch_event_csr_matvec_transpose(ct, values, indices, indptr, events, *,
                                      batch_size, shape, transpose):
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices.")

  if ad.is_undefined_primal(events):
    ct_events = (
      ad.Zero(events.aval) if type(ct) is ad.Zero else
      _batch_csr_matvec(ct, indices, indptr, values,
                        shape=shape, transpose=not transpose)
    )
    return values, indices, indptr, ct_events
  else:
    if values.aval.shape[1] == 1:  # scalar
      temp = event_csr_matvec_batching_p.bind(jnp.ones((1, 1)), indices, indptr, events,
                                              batch_size=batch_size, shape=shape,
                                              transpose=transpose)
      ct_values = jax.vmap(jnp.inner)(ct, temp)
    else:  # heterogeneous values
      if type(ct) is ad.Zero:
        ct_values = ad.Zero(values.aval)
      else:

        def _f(ct, indices, indptr, events, *, transpose):
          row, col = csr_to_coo(indices, indptr)
          ct_values = events[row] * ct[col] if transpose else events[col] * ct[row]
          return ct_values

        f = jax.vmap(partial(_f, transpose=transpose),
                     in_axes=(0,
                              0 if indices.shape[0] > 1 else None,
                              0 if indptr.shape[0] > 1 else None,
                              0 if events.shape[0] > 1 else None))
        ct_values = f(ct,
                      indices if indices.shape[0] > 1 else indices[0],
                      indptr if indptr.shape[0] > 1 else indptr[0],
                      events if events.shape[0] > 1 else events[0])
    return ct_values, indices, indptr, events


event_csr_matvec_batching_p = Primitive('event_csr_matvec_batching')
event_csr_matvec_batching_p.def_abstract_eval(_batch_event_csr_matvec_abstract)
event_csr_matvec_batching_p.def_impl(partial(xla.apply_primitive, event_csr_matvec_batching_p))
xla.backend_specific_translations['cpu'][event_csr_matvec_batching_p] = _batch_event_csr_matvec_cpu_translation
ad.defjvp(event_csr_matvec_batching_p, _batch_event_csr_matvec_jvp_values,
          None, None, _batch_event_csr_matvec_jvp_events)
ad.primitive_transposes[event_csr_matvec_batching_p] = _batch_event_csr_matvec_transpose


# operator for `event_csr_matvec` #
# ------------------------------- #


def _event_csr_matvec_abstract(values, indices, indptr, events, *, shape, transpose=False):
  return ShapedArray(dtype=values.dtype, shape=(shape[1] if transpose else shape[0],))


@numba.njit(fastmath=True)
def _event_csr_matvec_transpose_numba_imp(outs, ins):
  res_val = outs
  res_val.fill(0)
  values, indices, indptr, events, shape, _ = ins
  if values.shape[0] > 1:  # heter
    for row_i in range(shape[0]):
      if events[row_i]:
        for j in range(indptr[row_i], indptr[row_i + 1]):
          col_i = indices[j]
          res_val[col_i] += values[j]

  else:  # homo
    values = values[0]
    for row_i in range(shape[0]):
      if events[row_i]:
        for j in range(indptr[row_i], indptr[row_i + 1]):
          col_i = indices[j]
          res_val[col_i] += values


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _event_csr_matvec_numba_imp(outs, ins):
  res_val = outs
  res_val.fill(0)
  values, indices, indptr, events, shape, _ = ins

  if values.shape[0] > 1:  # heter
    for row_i in range(shape[0]):
      r = 0.
      for j in range(indptr[row_i], indptr[row_i + 1]):
        col_i = indices[j]
        if events[col_i]:
          r += values[j]
      res_val[row_i] = r

  else:  # homo
    values = values[0]
    for row_i in numba.prange(shape[0]):
      r = 0.
      for j in range(indptr[row_i], indptr[row_i + 1]):
        col_i = indices[j]
        if events[col_i]:
          r += values
      res_val[row_i] = r


def _event_csr_matvec_cpu_translation(c, values, indices, indptr, events, *, shape, transpose):
  inputs = (values, indices, indptr, events)
  description = dict(shape=shape, transpose=transpose)
  if transpose:
    name, inputs, in_layouts, out_layouts = compile_cpu_signature_with_numba(
      c,
      _event_csr_matvec_transpose_numba_imp,
      abs_eval_fn=_event_csr_matvec_abstract,
      multiple_results=False,
      inputs=inputs,
      description=description
    )
  else:
    name, inputs, in_layouts, out_layouts = compile_cpu_signature_with_numba(
      c,
      _event_csr_matvec_numba_imp,
      abs_eval_fn=_event_csr_matvec_abstract,
      multiple_results=False,
      inputs=inputs,
      description=description
    )
  return xla_client.ops.CustomCallWithLayout(
    c, name,
    operands=inputs,
    operand_shapes_with_layout=in_layouts,
    shape_with_layout=out_layouts,
  )


def _event_csr_matvec_gpu_translation(c, data, indices, indptr, vector, *, shape, transpose):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_csr_matvec_p.name)

  data_shape = c.get_shape(data)
  vec_shape = c.get_shape(vector)

  if data_shape.element_type() == jnp.float32:
    type_name = b'_float'
  elif data_shape.element_type() == jnp.float64:
    type_name = b'_double'
  else:
    raise ValueError

  data_name = b'_homo' if data_shape.dimensions() == (1,) else b'_heter'
  if vec_shape.element_type() == jnp.bool_:
    vec_type = b'_bool'
  else:
    if vec_shape.element_type() != data_shape.element_type():
      raise ValueError
    vec_type = type_name

  opaque = gpu_ops.build_twouint_onebool_descriptor(shape[0], shape[1], transpose)
  return xla_client.ops.CustomCallWithLayout(
    c,
    b'event_csr_matvec' + data_name + type_name + vec_type,
    operands=(data, indices, indptr, vector),
    operand_shapes_with_layout=(c.get_shape(data),
                                c.get_shape(indices),
                                c.get_shape(indptr),
                                c.get_shape(vector)),
    shape_with_layout=xla_client.Shape.array_shape(data_shape.element_type(),
                                                   (shape[1] if transpose else shape[0],),
                                                   (0,)),
    opaque=opaque,
  )


def _event_csr_matvec_batching_rule(args, axes, *, shape, transpose):
  batch_size = 0
  args_processed = []
  for arg, axis in zip(args, axes):
    if axis is None:
      arg = jnp.expand_dims(jnp.atleast_1d(arg), 0)
    else:
      batch_size = arg.shape[axis]
      if axis > 0:
        arg = jnp.moveaxis(arg, axis, 0)
    args_processed.append(arg)

  r = event_csr_matvec_batching_p.bind(*args_processed,
                                       batch_size=batch_size,
                                       shape=shape,
                                       transpose=transpose)
  return r, 0


def _event_csr_matvec_jvp_values(values_dot, values, indices, indptr, events, *, shape, transpose):
  return csrmv(values_dot, indices, indptr, events, shape=shape, transpose=transpose)


def _event_csr_matvec_jvp_events(events_dot, values, indices, indptr, events, *, shape, transpose):
  return normal_csrmv(values, indices, indptr, events_dot, shape=shape, transpose=transpose)


def _event_csr_matvec_transpose(ct, values, indices, indptr, events, *, shape, transpose):
  if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
    raise ValueError("Cannot transpose with respect to sparse indices.")
  if ad.is_undefined_primal(events):
    ct_events = normal_csrmv(values, indices, indptr, ct, shape=shape, transpose=not transpose)
    return values, indices, indptr, (ad.Zero(events) if type(ct) is ad.Zero else ct_events)
  else:
    if type(ct) is ad.Zero:
      ct_values = ad.Zero(values)
    else:
      if values.aval.shape[0] == 1:  # scalar
        ct_values = csrmv(jnp.ones(1), indices, indptr, events, shape=shape, transpose=transpose)
        ct_values = jnp.inner(ct, ct_values)
      else:  # heterogeneous values
        row, col = csr_to_coo(indices, indptr)
        ct_values = events[row] * ct[col] if transpose else events[col] * ct[row]
    return ct_values, indices, indptr, events


event_csr_matvec_p = Primitive('event_csr_matvec')
event_csr_matvec_p.def_abstract_eval(_event_csr_matvec_abstract)
event_csr_matvec_p.def_impl(partial(xla.apply_primitive, event_csr_matvec_p))
xla.backend_specific_translations['cpu'][event_csr_matvec_p] = _event_csr_matvec_cpu_translation
xla.backend_specific_translations['gpu'][event_csr_matvec_p] = _event_csr_matvec_gpu_translation
ad.defjvp(event_csr_matvec_p, _event_csr_matvec_jvp_values, None, None, _event_csr_matvec_jvp_events)
ad.primitive_transposes[event_csr_matvec_p] = _event_csr_matvec_transpose
register_general_batching(event_csr_matvec_p)
# batching.primitive_batchers[event_csr_matvec_p] = _event_csr_matvec_batching_rule
