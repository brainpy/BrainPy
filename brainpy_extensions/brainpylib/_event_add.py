# -*- coding: utf-8 -*-

__all__ = [
  'event_add',
  'event_add2',
]

from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import core, dtypes
from jax.interpreters import xla
from jax.lib import xla_client

x_shape = xla_client.Shape.array_shape
x_ops = xla_client.ops

# Register the CPU XLA custom calls
from . import cpu_ops
for _name, _value in cpu_ops.registrations().items():
  xla_client.register_cpu_custom_call_target(_name, _value)


_event_add_prim = core.Primitive("event_add")


def event_add(events, pre2post, post_num, value):
  # output value
  value = jnp.asarray(value)
  out = jnp.zeros(post_num, dtype=value.dtype)
  # connections
  indices, indptr = pre2post
  assert len(events) + 1 == len(indptr)
  return _event_add_prim.bind(events, indices, indptr, value, out)


def _event_add_abstract(events, indices, indptr, value, out):
  dtype1 = dtypes.canonicalize_dtype(indices.dtype)
  dtype2 = dtypes.canonicalize_dtype(indptr.dtype)
  assert dtype1 in [np.uint32, np.uint64]
  assert dtype2 in [np.uint32, np.uint64]
  assert dtype1 == dtype2
  assert dtypes.canonicalize_dtype(events.dtype) == np.bool_
  return out


_event_add_prim.def_abstract_eval(_event_add_abstract)
_event_add_prim.def_impl(partial(xla.apply_primitive, _event_add_prim))


def _event_add_translation(c, events, indices, indptr, value, out, *, platform="cpu"):
  # The event shape
  events_shape = c.get_shape(events)
  events_dim = events_shape.dimensions()
  _events_shape = x_shape(events_shape.element_type(), events_dim, (0,))

  # The pre_size shape
  pre_size = np.array(events_dim[0], dtype=np.uint32)
  _pre_shape = x_shape(np.dtype(np.uint32), (), ())

  # The indices shape
  indices_shape = c.get_shape(indices)
  Itype = indices_shape.element_type()
  _indices_shape = x_shape(Itype, indices_shape.dimensions(), (0,))

  # The indptr shape
  _indptr_shape = x_shape(Itype, c.get_shape(indptr).dimensions(), (0,))

  # The value shape
  Ftype = c.get_shape(value).element_type()
  _value_shape = x_shape(Ftype, (), ())

  # The output value shape
  out_shape = c.get_shape(out)
  _out_shape = x_shape(Ftype, out_shape.dimensions(), (0,))

  # We dispatch a different call depending on the dtype
  if Ftype == np.float32:
    if Itype == np.uint32:
      op_name = platform.encode() + b"_event_add_f32_i32"
    elif Itype == np.uint64:
      op_name = platform.encode() + b"_event_add_f32_i64"
    else:
      raise NotImplementedError
  elif Ftype == np.float64:
    if Itype == np.uint32:
      op_name = platform.encode() + b"_event_add_f64_i32"
    elif Itype == np.uint64:
      op_name = platform.encode() + b"_event_add_f64_i64"
    else:
      raise NotImplementedError
  else: raise NotImplementedError

  # And then the following is what changes between the GPU and CPU
  if platform == "cpu":
    # On the CPU, we pass the size of the data as a the first input argument
    return x_ops.CustomCallWithLayout(
      c,  # builder
      op_name,  # call_target_name
      operands=(x_ops.ConstantLiteral(c, pre_size), events, indices, indptr, value),  # The inputs
      operand_shapes_with_layout=(
        _pre_shape, _events_shape, _indices_shape, _indptr_shape, _value_shape),  # The input shapes
      shape_with_layout=_out_shape,  # The output shapes
    )
  elif platform == 'gpu':
    pass

  raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


# Connect the XLA translation rules for JIT compilation
xla.backend_specific_translations["cpu"][_event_add_prim] = partial(_event_add_translation, platform="cpu")


_event_add2_prim = core.Primitive("event_add2")


def event_add2(events, pre_ids, post_ids, post_num, value):
  # output value
  value = jnp.asarray(value)
  out = jnp.zeros(post_num, dtype=value.dtype)
  # connections
  assert len(pre_ids) == len(post_ids)
  return _event_add2_prim.bind(events, pre_ids, post_ids, value, out)



def _event_add2_abstract(events, pre_ids, post_ids, value, out):
  dtype1 = dtypes.canonicalize_dtype(pre_ids.dtype)
  dtype2 = dtypes.canonicalize_dtype(post_ids.dtype)
  assert dtype1 in [np.uint32, np.uint64]
  assert dtype2 in [np.uint32, np.uint64]
  assert dtype1 == dtype2
  assert dtypes.canonicalize_dtype(events.dtype) == np.bool_
  return out


_event_add2_prim.def_abstract_eval(_event_add2_abstract)
_event_add2_prim.def_impl(partial(xla.apply_primitive, _event_add2_prim))


def _event_add2_translation(c, events, pre_ids, post_ids, value, out, *, platform="cpu"):
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
      op_name = platform.encode() + b"_event_add2_f32_i32"
    elif Itype == np.uint64:
      op_name = platform.encode() + b"_event_add2_f32_i64"
    else:
      raise NotImplementedError
  elif Ftype == np.float64:
    if Itype == np.uint32:
      op_name = platform.encode() + b"_event_add2_f64_i32"
    elif Itype == np.uint64:
      op_name = platform.encode() + b"_event_add2_f64_i64"
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


xla.backend_specific_translations["cpu"][_event_add2_prim] = partial(_event_add2_translation, platform="cpu")
