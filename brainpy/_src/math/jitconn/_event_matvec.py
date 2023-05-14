# -*- coding: utf-8 -*-

from functools import partial
from typing import Tuple, Optional

import jax
import numpy as np
from jax import numpy as jnp, dtypes
from jax.core import ShapedArray, Primitive
from jax.interpreters import xla, ad
from jax.lib import xla_client

from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.op_registers import register_general_batching
from brainpy.errors import GPUOperatorNotFound
from ._matvec import (
  matvec_prob_homo_p,
  matvec_prob_uniform_p,
  matvec_prob_normal_p,
  mv_prob_homo,
  mv_prob_uniform,
  mv_prob_normal
)

try:
  from brainpylib import gpu_ops
except ImportError:
  gpu_ops = None

__all__ = [
  'event_mv_prob_homo',
  'event_mv_prob_uniform',
  'event_mv_prob_normal',
]


def event_mv_prob_homo(
    events: jax.Array,
    weight: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  events = as_jax(events)
  weight = jnp.atleast_1d(as_jax(weight))
  conn_prob = jnp.atleast_1d(as_jax(conn_prob))
  clen = jnp.asarray(jnp.ceil(1 / conn_prob) * 2 - 1, dtype=jnp.int32)
  with jax.ensure_compile_time_eval():
    if seed is None:
      seed = int(np.random.randint(0, int(1e8)))
  r = event_matvec_prob_homo_p.bind(events,
                                    weight,
                                    clen,
                                    shape=shape,
                                    seed=seed,
                                    transpose=transpose,
                                    outdim_parallel=outdim_parallel)[0]
  return r


event_mv_prob_homo.__doc__ = mv_prob_homo.__doc__


def event_mv_prob_uniform(
    events: jax.Array,
    w_low: float,
    w_high: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  events = as_jax(events)

  w_low = jnp.atleast_1d(as_jax(w_low))
  w_high = jnp.atleast_1d(as_jax(w_high))
  conn_prob = jnp.atleast_1d(as_jax(conn_prob))
  clen = jnp.asarray(jnp.ceil(1 / conn_prob) * 2 - 1, dtype=jnp.int32)
  with jax.ensure_compile_time_eval():
    if seed is None:
      seed = int(np.random.randint(0, int(1e8)))
  return event_matvec_prob_uniform_p.bind(events,
                                          w_low,
                                          w_high,
                                          clen,
                                          shape=shape,
                                          seed=seed,
                                          transpose=transpose,
                                          outdim_parallel=outdim_parallel)[0]


event_mv_prob_uniform.__doc__ = mv_prob_uniform.__doc__


def event_mv_prob_normal(
    events: jax.Array,
    w_mu: float,
    w_sigma: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
  events = as_jax(events)

  w_mu = jnp.atleast_1d(as_jax(w_mu))
  w_sigma = jnp.atleast_1d(as_jax(w_sigma))
  conn_prob = jnp.atleast_1d(as_jax(conn_prob))
  clen = jnp.asarray(jnp.ceil(1 / conn_prob) * 2 - 1, dtype=jnp.int32)
  with jax.ensure_compile_time_eval():
    if seed is None:
      seed = int(np.random.randint(0, int(1e8)))
  return event_matvec_prob_normal_p.bind(events,
                                         w_mu,
                                         w_sigma,
                                         clen,
                                         shape=shape,
                                         seed=seed,
                                         transpose=transpose,
                                         outdim_parallel=outdim_parallel)[0]


event_mv_prob_normal.__doc__ = mv_prob_normal.__doc__


def _event_matvec_prob_homo_abstract(
    events, weight, clen, *, shape, seed, transpose, outdim_parallel
):
  if events.ndim != 1:
    raise ValueError('events should be a 1D vector.')
  if len(shape) != 2:
    raise ValueError('shape should be a length-2 tuple.')
  if not np.isscalar(seed):
    raise ValueError('seed must be a scalar.')
  if not isinstance(outdim_parallel, bool):
    raise ValueError('outdim_parallel must be boolean value.')
  if not isinstance(transpose, bool):
    raise ValueError('transpose must be boolean value.')
  if clen.ndim != 1:
    raise ValueError('conn_prob must be a 1D scalar.')
  if weight.ndim != 1:
    raise ValueError('weight must be a 1D scalar.')

  if transpose:
    if events.shape[0] != shape[0]:
      raise ValueError(f'Shape mismatch, vec ({events.shape[0]},) @ mat {shape}.')
  else:
    if events.shape[0] != shape[1]:
      raise ValueError(f'Shape mismatch, mat {shape} @ vec ({events.shape[0]},).')
  out = ShapedArray(dtype=(dtypes.canonicalize_dtype(float)
                           if (events.dtype == jnp.bool_) else events.dtype),
                    shape=(shape[1] if transpose else shape[0],))
  return [out]


def _event_matvec_prob_homo_cpu_translation(
    c, events, weight, clen, *, shape, seed, transpose, outdim_parallel
):
  n_row, n_col = (shape[1], shape[0]) if transpose else shape
  event_shape = c.get_shape(events)
  if event_shape.element_type() == jnp.bool_:
    event_type = b'_bool'
    out_dtype = dtypes.canonicalize_dtype(float)
    type_name = b'_float' if out_dtype == jnp.float32 else b'_double'
  else:
    out_dtype = event_shape.element_type()
    event_type = b'_float' if out_dtype == jnp.float32 else b'_double'
    type_name = event_type

  if outdim_parallel:
    fn = b'cpu_event_matvec_prob_homo' + type_name + event_type
  else:
    fn = b'cpu_event_matvec_atomic_prob_homo' + type_name + event_type

  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(events,
              weight,
              clen,
              xla_client.ops.ConstantLiteral(c, seed),
              xla_client.ops.ConstantLiteral(c, n_row),
              xla_client.ops.ConstantLiteral(c, n_col)),
    operand_shapes_with_layout=(c.get_shape(events),
                                c.get_shape(weight),
                                c.get_shape(clen),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ())),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
  )


def _event_matvec_prob_homo_gpu_translation(
    c, events, weight, clen, *, shape, seed, transpose, outdim_parallel
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_matvec_prob_homo_p.name)

  event_shape = c.get_shape(events)
  if event_shape.element_type() == jnp.bool_:
    event_type = b'_bool'
    out_dtype = dtypes.canonicalize_dtype(float)
    type_name = b'_float' if out_dtype == jnp.float32 else b'_double'
  else:
    out_dtype = event_shape.element_type()
    event_type = b'_float' if out_dtype == jnp.float32 else b'_double'
    type_name = event_type

  opaque = gpu_ops.build_triple_size_descriptor(shape[1] if transpose else shape[0],
                                                shape[0] if transpose else shape[1],
                                                seed)

  if outdim_parallel:
    fn = b'gpu_event_matvec_prob_homo_v2' + type_name + event_type
  else:
    fn = b'gpu_event_matvec_atomic_prob_homo_v2' + type_name + event_type

  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(events, weight, clen),
    operand_shapes_with_layout=(c.get_shape(events),
                                c.get_shape(weight),
                                c.get_shape(clen)),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
    opaque=opaque,
  )


def _event_matvec_prob_homo_jvp(
    primals, tangents, *, shape, seed, transpose, outdim_parallel
):
  events, weight, clen = primals
  event_dot, = tangents
  r = event_matvec_prob_homo_p.bind(events,
                                    weight,
                                    clen,
                                    shape=shape,
                                    seed=seed,
                                    transpose=transpose,
                                    outdim_parallel=outdim_parallel)
  dr = matvec_prob_homo_p.bind(event_dot,
                               weight,
                               clen,
                               shape=shape,
                               seed=seed,
                               transpose=transpose,
                               outdim_parallel=outdim_parallel)
  return r, dr


def _event_matvec_prob_homo_transpose(
    ct, events, weight, clen, *, shape, seed, transpose, outdim_parallel
):
  ct_event = matvec_prob_homo_p.bind(ct,
                                     weight,
                                     clen,
                                     seed=seed,
                                     shape=shape,
                                     transpose=not transpose,
                                     outdim_parallel=not outdim_parallel)
  return ct_event


event_matvec_prob_homo_p = Primitive('event_matvec_prob_conn_homo_weight')
event_matvec_prob_homo_p.multiple_results = True
event_matvec_prob_homo_p.def_abstract_eval(_event_matvec_prob_homo_abstract)
event_matvec_prob_homo_p.def_impl(partial(xla.apply_primitive, event_matvec_prob_homo_p))
xla.backend_specific_translations['cpu'][event_matvec_prob_homo_p] = _event_matvec_prob_homo_cpu_translation
xla.backend_specific_translations['gpu'][event_matvec_prob_homo_p] = _event_matvec_prob_homo_gpu_translation
ad.primitive_jvps[event_matvec_prob_homo_p] = _event_matvec_prob_homo_jvp
ad.primitive_transposes[event_matvec_prob_homo_p] = _event_matvec_prob_homo_transpose
register_general_batching(event_matvec_prob_homo_p)


def _event_matvec_prob_uniform_abstract(
    events, w_low, w_high, clen, *, shape, seed, transpose, outdim_parallel
):
  if events.ndim != 1:
    raise ValueError('events should be a 1D vector.')
  if len(shape) != 2:
    raise ValueError('shape should be a length-2 tuple.')
  if w_low.ndim != 1:
    raise ValueError('w_low must be a 1D scalar.')
  if w_high.ndim != 1:
    raise ValueError('w_high must be a 1D scalar.')
  if clen.ndim != 1:
    raise ValueError('clen must be a 1D scalar.')
  if not np.isscalar(seed):
    raise ValueError('seed must be a scalar.')
  if not isinstance(transpose, bool):
    raise ValueError('transpose must be a boolean value.')
  if not isinstance(outdim_parallel, bool):
    raise ValueError('outdim_parallel must be a boolean value.')
  assert w_low.dtype == w_high.dtype

  if transpose:
    if events.shape[0] != shape[0]:
      raise ValueError(f'Shape mismatch, vec ({events.shape[0]},) @ mat {shape}.')
  else:
    if events.shape[0] != shape[1]:
      raise ValueError(f'Shape mismatch, mat {shape} @ vec ({events.shape[0]},).')

  out = ShapedArray(dtype=(dtypes.canonicalize_dtype(float)
                           if events.dtype == jnp.bool_ else events.dtype),
                    shape=(shape[1] if transpose else shape[0],))
  return [out]


def _event_matvec_prob_uniform_cpu_translation(
    c, events, w_low, w_high, clen, *, shape, seed, transpose, outdim_parallel
):
  n_row, n_col = (shape[1], shape[0]) if transpose else shape

  event_shape = c.get_shape(events)
  if event_shape.element_type() == jnp.bool_:
    event_type = b'_bool'
    out_dtype = dtypes.canonicalize_dtype(float)
    type_name = b'_float' if (out_dtype == jnp.float32) else b'_double'
  else:
    out_dtype = event_shape.element_type()
    event_type = b'_float' if (out_dtype == jnp.float32) else b'_double'
    type_name = event_type

  if outdim_parallel:
    fn = b'cpu_event_matvec_prob_uniform' + type_name + event_type
  else:
    fn = b'cpu_event_matvec_atomic_prob_uniform' + type_name + event_type
  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(events,
              clen,
              w_low,
              w_high,
              xla_client.ops.ConstantLiteral(c, seed),
              xla_client.ops.ConstantLiteral(c, n_row),
              xla_client.ops.ConstantLiteral(c, n_col)),
    operand_shapes_with_layout=(c.get_shape(events),
                                c.get_shape(clen),
                                c.get_shape(w_low),
                                c.get_shape(w_high),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ())),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
  )


def _event_matvec_prob_uniform_gpu_translation(
    c, events, w_low, w_high, clen, *, shape, seed, transpose, outdim_parallel
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_matvec_prob_homo_p.name)

  event_shape = c.get_shape(events)
  if event_shape.element_type() == jnp.bool_:
    event_type = b'_bool'
    out_dtype = dtypes.canonicalize_dtype(float)
    type_name = b'_float' if out_dtype == jnp.float32 else b'_double'
  else:
    out_dtype = event_shape.element_type()
    event_type = b'_float' if out_dtype == jnp.float32 else b'_double'
    type_name = event_type

  opaque = gpu_ops.build_triple_size_descriptor(shape[1] if transpose else shape[0],
                                                shape[0] if transpose else shape[1],
                                                seed)
  if outdim_parallel:
    fn = b'gpu_event_matvec_prob_uniform_v2' + type_name + event_type
  else:
    fn = b'gpu_event_matvec_atomic_prob_uniform_v2' + type_name + event_type
  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(events, w_low, w_high, clen),
    operand_shapes_with_layout=(c.get_shape(events),
                                c.get_shape(w_low),
                                c.get_shape(w_high),
                                c.get_shape(clen),),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
    opaque=opaque,
  )


def _event_matvec_prob_uniform_jvp(
    primals, tangents, *, shape, seed, transpose, outdim_parallel
):
  events, w_low, w_high, clen = primals
  events_dot, = tangents
  r = event_matvec_prob_uniform_p.bind(events,
                                       w_low,
                                       w_high,
                                       clen,
                                       shape=shape,
                                       seed=seed,
                                       outdim_parallel=outdim_parallel,
                                       transpose=transpose)
  r_dot = matvec_prob_uniform_p.bind(events_dot,
                                     w_low,
                                     w_high,
                                     clen,
                                     shape=shape,
                                     seed=seed,
                                     transpose=transpose,
                                     outdim_parallel=outdim_parallel,
                                     version='v2')
  return r, r_dot


def _event_matvec_prob_uniform_transpose(
    ct, events, w_low, w_high, clen, *, shape, seed, transpose, outdim_parallel
):
  return matvec_prob_uniform_p.bind(ct,
                                    w_low,
                                    w_high,
                                    clen,
                                    seed=seed,
                                    shape=shape,
                                    transpose=not transpose,
                                    outdim_parallel=not outdim_parallel,
                                    version='v2')


event_matvec_prob_uniform_p = Primitive('event_matvec_prob_uniform')
event_matvec_prob_uniform_p.multiple_results = True
event_matvec_prob_uniform_p.def_abstract_eval(_event_matvec_prob_uniform_abstract)
event_matvec_prob_uniform_p.def_impl(partial(xla.apply_primitive, event_matvec_prob_uniform_p))
xla.backend_specific_translations['cpu'][event_matvec_prob_uniform_p] = _event_matvec_prob_uniform_cpu_translation
xla.backend_specific_translations['gpu'][event_matvec_prob_uniform_p] = _event_matvec_prob_uniform_gpu_translation
register_general_batching(event_matvec_prob_uniform_p)
ad.primitive_jvps[event_matvec_prob_uniform_p] = _event_matvec_prob_uniform_jvp
ad.primitive_transposes[event_matvec_prob_uniform_p] = _event_matvec_prob_uniform_transpose


def _event_matvec_prob_normal_abstract(
    events, w_mu, w_sigma, clen, *, shape, seed, transpose, outdim_parallel
):
  if w_mu.ndim != 1:
    raise ValueError('w_mu should be a 1D scalar.')
  if w_sigma.ndim != 1:
    raise ValueError('w_sigma should be a 1D scalar.')
  if clen.ndim != 1:
    raise ValueError('clen should be a 1D scalar.')
  if np.ndim(events) != 1:
    raise ValueError('events should be a 1D vector.')
  if len(shape) != 2:
    raise ValueError('shape should be a length-2 tuple.')
  if not isinstance(transpose, bool):
    raise ValueError('transpose must be a boolean value.')
  if not isinstance(outdim_parallel, bool):
    raise ValueError('outdim_parallel must be a boolean value.')
  if transpose:
    if events.shape[0] != shape[0]:
      raise ValueError(f'Shape mismatch, vec ({events.shape[0]},) @ mat {shape}.')
  else:
    if events.shape[0] != shape[1]:
      raise ValueError(f'Shape mismatch, mat {shape} @ vec ({events.shape[0]},).')
  if not np.isscalar(seed):
    raise ValueError('seed must be a scalar.')

  out = ShapedArray(dtype=(dtypes.canonicalize_dtype(float)
                           if events.dtype == jnp.bool_ else events.dtype),
                    shape=(shape[1] if transpose else shape[0],))
  return [out]


def _event_matvec_prob_normal_cpu_translation(
    c, events, w_mu, w_sigma, clen, *, shape, seed, transpose, outdim_parallel
):
  n_row, n_col = (shape[1], shape[0]) if transpose else shape

  event_shape = c.get_shape(events)
  if event_shape.element_type() == jnp.bool_:
    event_type = b'_bool'
    out_dtype = dtypes.canonicalize_dtype(float)
    type_name = b'_float' if out_dtype == jnp.float32 else b'_double'
  else:
    out_dtype = event_shape.element_type()
    event_type = b'_float' if out_dtype == jnp.float32 else b'_double'
    type_name = event_type

  if outdim_parallel:
    fn = b'cpu_event_matvec_prob_normal' + type_name + event_type
  else:
    fn = b'cpu_event_matvec_atomic_prob_normal' + type_name + event_type
  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(events,
              clen,
              w_mu,
              w_sigma,
              xla_client.ops.ConstantLiteral(c, seed),
              xla_client.ops.ConstantLiteral(c, n_row),
              xla_client.ops.ConstantLiteral(c, n_col)),
    operand_shapes_with_layout=(c.get_shape(events),
                                c.get_shape(clen),
                                c.get_shape(w_mu),
                                c.get_shape(w_sigma),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ()),
                                xla_client.Shape.array_shape(np.dtype(np.uint32), (), ())),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
  )


def _event_matvec_prob_normal_gpu_translation(
    c, events, w_mu, w_sigma, clen, *, shape, seed, transpose, outdim_parallel
):
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_matvec_prob_homo_p.name)

  event_shape = c.get_shape(events)
  if event_shape.element_type() == jnp.bool_:
    event_type = b'_bool'
    out_dtype = dtypes.canonicalize_dtype(float)
    type_name = b'_float' if out_dtype == jnp.float32 else b'_double'
  else:
    out_dtype = event_shape.element_type()
    event_type = b'_float' if out_dtype == jnp.float32 else b'_double'
    type_name = event_type
  opaque = gpu_ops.build_triple_size_descriptor(shape[1] if transpose else shape[0],
                                                shape[0] if transpose else shape[1],
                                                seed)
  if outdim_parallel:
    fn = b'gpu_event_matvec_prob_normal_v2' + type_name + event_type
  else:
    fn = b'gpu_event_matvec_atomic_prob_normal_v2' + type_name + event_type
  return xla_client.ops.CustomCallWithLayout(
    c,
    fn,
    operands=(events, w_mu, w_sigma, clen),
    operand_shapes_with_layout=(c.get_shape(events),
                                c.get_shape(w_mu),
                                c.get_shape(w_sigma),
                                c.get_shape(clen),),
    shape_with_layout=xla_client.Shape.tuple_shape(
      (
        xla_client.Shape.array_shape(out_dtype, (shape[1] if transpose else shape[0],), (0,)),
      )
    ),
    opaque=opaque,
  )


def _event_matvec_prob_normal_jvp(
    primals, tangents, *, w_mu, w_sigma, conn_prob, shape, seed, transpose, outdim_parallel
):
  events, = primals
  events_dot, = tangents
  r = event_matvec_prob_normal_p.bind(events,
                                      w_mu=w_mu,
                                      w_sigma=w_sigma,
                                      conn_prob=conn_prob,
                                      shape=shape,
                                      seed=seed,
                                      transpose=transpose,
                                      outdim_parallel=outdim_parallel)
  r_dot = matvec_prob_normal_p.bind(events_dot,
                                    w_mu=w_mu,
                                    w_sigma=w_sigma,
                                    conn_prob=conn_prob,
                                    shape=shape,
                                    seed=seed,
                                    transpose=transpose,
                                    outdim_parallel=outdim_parallel,
                                    version='v2')
  return r, r_dot


def _event_matvec_prob_normal_transpose(
    ct, events, *, w_mu, w_sigma, conn_prob, shape, seed, transpose, outdim_parallel
):
  return matvec_prob_normal_p.bind(ct,
                                   w_mu=w_mu,
                                   w_sigma=w_sigma,
                                   conn_prob=conn_prob,
                                   seed=seed,
                                   shape=shape,
                                   transpose=not transpose,
                                   outdim_parallel=not outdim_parallel,
                                   version='v2')


event_matvec_prob_normal_p = Primitive('event_matvec_prob_normal')
event_matvec_prob_normal_p.multiple_results = True
event_matvec_prob_normal_p.def_abstract_eval(_event_matvec_prob_normal_abstract)
event_matvec_prob_normal_p.def_impl(partial(xla.apply_primitive, event_matvec_prob_normal_p))
xla.backend_specific_translations['cpu'][event_matvec_prob_normal_p] = _event_matvec_prob_normal_cpu_translation
xla.backend_specific_translations['gpu'][event_matvec_prob_normal_p] = _event_matvec_prob_normal_gpu_translation
register_general_batching(event_matvec_prob_normal_p)
ad.primitive_jvps[event_matvec_prob_normal_p] = _event_matvec_prob_normal_jvp
ad.primitive_transposes[event_matvec_prob_normal_p] = _event_matvec_prob_normal_transpose
