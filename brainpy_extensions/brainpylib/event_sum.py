# -*- coding: utf-8 -*-

__all__ = [
  'event_sum',
]

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import core, dtypes
from jax.interpreters import xla
from jax.lib import xla_client, xla_bridge

try:
  from . import gpu_ops
except ImportError:
  gpu_ops = None

x_shape = xla_client.Shape.array_shape
x_ops = xla_client.ops

_event_sum_prim = core.Primitive("event_sum")


def event_sum(events, pre2post, post_num, values):
  # events
  if events.dtype != jnp.bool_:
    raise ValueError(f'"events" must be a vector of bool, while we got {events.dtype}')

  # connections
  indices, indptr = pre2post
  if len(events) + 1 != len(indptr):
    raise ValueError(f'The length of "events" must be equal to "len(pre2post[1]) - 1", '
                     f'while we get: {len(events)} + 1 != {len(indptr)}')
  if indices.dtype != indptr.dtype:
    raise ValueError(f"The dtype of pre2post[0] must be equal to that of pre2post[1], "
                     f"while we got {(indices.dtype, indptr.dtype)}")
  if indices.dtype not in [jnp.uint32, jnp.uint64]:
    raise ValueError(f'The dtype of pre2post must be uint32 or uint64, while we got {indices.dtype}')

  # output value
  values = jnp.asarray(values)
  if values.dtype not in [jnp.float32, jnp.float64]:
    raise ValueError(f'The dtype of "values" must be float32 or float64, while we got {values.dtype}.')
  if values.size not in [1, indices.size]:
    raise ValueError(f'The size of "values" must be 1 (a scalar) or len(pre2post[0]) (a vector), '
                     f'while we got {values.size} != 1 != {indices.size}')
  out = jnp.zeros(post_num, dtype=values.dtype)

  # TODO: GPU operator, method 1
  if xla_bridge.get_backend().platform == 'gpu':
    indptr = jnp.repeat(jnp.arange(indptr.shape[0] - 1, dtype=indptr.dtype), jnp.diff(indptr))

  # TODO: GPU operator, method 2
  if xla_bridge.get_backend().platform != 'cpu':
    indptr = jnp.repeat(jnp.arange(indptr.shape[0] - 1, dtype=indptr.dtype), jnp.diff(indptr))
    out = out.at[indices].add(events[indptr] * values)
    return out

  # bind operator
  return _event_sum_prim.bind(events, indices, indptr, values, out)


def _event_sum_abstract(events, indices, indptr, values, out):
  return out


_event_sum_prim.def_abstract_eval(_event_sum_abstract)
_event_sum_prim.def_impl(partial(xla.apply_primitive, _event_sum_prim))


def _event_sum_translation(c, events, indices, indptr, values, out, *, platform="cpu"):
  # The output value shape
  out_shape = c.get_shape(out)

  # The event shape
  events_shape = c.get_shape(events)

  # The pre/post shape
  pre_size = np.array(events_shape.dimensions()[0], dtype=np.uint32)
  post_size = np.array(out_shape.dimensions()[0], dtype=np.uint32)
  _pre_shape = x_shape(np.dtype(np.uint32), (), ())
  _post_shape = x_shape(np.dtype(np.uint32), (), ())

  # The indices shape
  indices_shape = c.get_shape(indices)
  Itype = indices_shape.element_type()
  assert Itype in [np.uint32, np.uint64]

  # The value shape
  values_shape = c.get_shape(values)
  Ftype = values_shape.element_type()
  assert Ftype in [np.float32, np.float64]
  values_dim = values_shape.dimensions()
  if len(values_dim) != 0:
    assert values_dim == indices_shape.dimensions()

  # We dispatch a different call depending on the dtype
  f_type = b'_f32' if Ftype == np.float32 else b'_f64'
  i_type = b'_i32' if Itype == np.uint32 else b'_i64'

  # And then the following is what changes between the GPU and CPU
  if platform == "cpu":
    v_type = b'_event_sum_homo' if len(values_dim) == 0 else b'_event_sum_heter'
    return x_ops.CustomCallWithLayout(
      c, platform.encode() + v_type + f_type + i_type,
      operands=(
        x_ops.ConstantLiteral(c, pre_size), x_ops.ConstantLiteral(c, post_size),
        events, indices, indptr, values
      ),
      operand_shapes_with_layout=(
        _pre_shape, _post_shape, c.get_shape(events),
        c.get_shape(indices), c.get_shape(indptr), c.get_shape(values)
      ),
      shape_with_layout=c.get_shape(out),
    )
  elif platform == 'gpu':
    if gpu_ops is None:
      raise ValueError('Cannot find compiled gpu wheels.')

    v_type = b'_event_sum2_homo' if len(values_dim) == 0 else b'_event_sum2_heter'
    opaque = gpu_ops.build_gpu_descriptor(pre_size, post_size)
    return x_ops.CustomCallWithLayout(
      c, platform.encode() + v_type + f_type + i_type,
      operands=(events, indptr, indices, values),
      operand_shapes_with_layout=(c.get_shape(events), c.get_shape(indptr),
                                  c.get_shape(indices), c.get_shape(values)),
      shape_with_layout=c.get_shape(out),
      opaque=opaque,
    )

  else:
    raise ValueError("Unsupported platform, we only support 'cpu' or 'gpu'")


xla.backend_specific_translations["cpu"][_event_sum_prim] = \
  partial(_event_sum_translation, platform="cpu")
xla.backend_specific_translations["gpu"][_event_sum_prim] = \
  partial(_event_sum_translation, platform="gpu")

# ---------------------------
#
# ---------------------------


_event_sum2_prim = core.Primitive("event_sum2")


def event_sum2(events, pre_ids, post_ids, post_num, value):
  # output value
  value = jnp.asarray(value)
  out = jnp.zeros(post_num, dtype=value.dtype)
  # connections
  assert len(pre_ids) == len(post_ids)
  return _event_sum2_prim.bind(events, pre_ids, post_ids, value, out)


def _event_sum2_abstract(events, pre_ids, post_ids, value, out):
  dtype1 = dtypes.canonicalize_dtype(pre_ids.dtype)
  dtype2 = dtypes.canonicalize_dtype(post_ids.dtype)
  assert dtype1 in [np.uint32, np.uint64]
  assert dtype2 in [np.uint32, np.uint64]
  assert dtype1 == dtype2
  assert dtypes.canonicalize_dtype(events.dtype) == np.bool_
  return out


_event_sum2_prim.def_abstract_eval(_event_sum2_abstract)
_event_sum2_prim.def_impl(partial(xla.apply_primitive, _event_sum2_prim))


def _event_sum2_translation(c, events, pre_ids, post_ids, value, out, *, platform="cpu"):
  # The event shape
  events_shape = c.get_shape(events)
  events_dim = events_shape.dimensions()
  _events_shape = x_shape(events_shape.element_type(), events_dim, (0,))

  # The post_ids shape
  pre_ids_shape = c.get_shape(pre_ids)
  Itype = pre_ids_shape.element_type()
  _pre_ids_shape = x_shape(Itype, pre_ids_shape.dimensions(), (0,))

  # The pre_size shape
  conn_size = np.array(pre_ids_shape.dimensions()[0], dtype=np.uint32)
  _conn_shape = x_shape(np.dtype(np.uint32), (), ())

  # The pre_slice shape
  _post_ids_shape = x_shape(Itype, c.get_shape(post_ids).dimensions(), (0,))

  # The value shape
  value_shape = c.get_shape(value)
  Ftype = value_shape.element_type()
  _value_shape = x_shape(Ftype, (), ())

  # The output value shape
  _out_shape = x_shape(Ftype, c.get_shape(out).dimensions(), (0,))

  # We dispatch a different call depending on the dtype
  if Ftype == np.float32:
    if Itype == np.uint32:
      op_name = platform.encode() + b"_event_sum2_f32_i32"
    elif Itype == np.uint64:
      op_name = platform.encode() + b"_event_sum2_f32_i64"
    else:
      raise NotImplementedError
  elif Ftype == np.float64:
    if Itype == np.uint32:
      op_name = platform.encode() + b"_event_sum2_f64_i32"
    elif Itype == np.uint64:
      op_name = platform.encode() + b"_event_sum2_f64_i64"
    else:
      raise NotImplementedError
  else:
    raise NotImplementedError(f"Unsupported dtype {Ftype}")

  # And then the following is what changes between the GPU and CPU
  if platform == "cpu":
    # On the CPU, we pass the size of the data as an input argument
    return x_ops.CustomCallWithLayout(
      c,  # builder
      op_name,  # call_target_name
      operands=(events,
                pre_ids,
                post_ids,
                x_ops.ConstantLiteral(c, conn_size),
                value),  # The inputs
      operand_shapes_with_layout=(_events_shape,
                                  _pre_ids_shape,
                                  _post_ids_shape,
                                  _conn_shape,
                                  _value_shape),  # The input shapes
      shape_with_layout=_out_shape,  # The output shapes
    )
  elif platform == 'gpu':
    pass

  raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


xla.backend_specific_translations["cpu"][_event_sum2_prim] = partial(_event_sum2_translation, platform="cpu")
