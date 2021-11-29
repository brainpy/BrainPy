# -*- coding: utf-8 -*-

__all__ = ["event_add_prim", "event_slice_add_prim"]

from functools import partial

import brainpy.math as bm
import jax.numpy as jnp
import numpy as np
from jax import core, dtypes
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla
from jax.lib import xla_client

# Register the CPU XLA custom calls
from brainpylib import cpu_ops

for _name, _value in cpu_ops.registrations().items():
  xla_client.register_cpu_custom_call_target(_name, _value)


# This function exposes the primitive to user code and this is the only
# public-facing function in this module
def event_add(events, conn_mat, value=1.):
  if isinstance(events, bm.JaxArray): events = events.value
  if isinstance(conn_mat, bm.JaxArray): conn_mat = conn_mat.value
  if isinstance(value, bm.JaxArray): value = value.value
  value = jnp.array(value)
  return event_add_prim.bind(events, conn_mat.flatten(), value)


event_add_prim = core.Primitive("event_add")


def event_slice_add(events, post_ids, pre_slice, post_num, value=1.):
  if isinstance(events, bm.JaxArray): events = events.value
  if isinstance(post_ids, bm.JaxArray): post_ids = post_ids.value
  if isinstance(pre_slice, bm.JaxArray): pre_slice = pre_slice.value
  pre_slice = pre_slice.flatten()
  value = jnp.array(value)
  out = jnp.zeros(post_num, dtype=value.dtype)
  return event_slice_add_prim.bind(events, post_ids, pre_slice, value, out)


event_slice_add_prim = core.Primitive("event_slice_add")


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _event_add_abstract(events, conn_mat, value):
  assert dtypes.canonicalize_dtype(events.dtype) == np.bool_
  assert dtypes.canonicalize_dtype(conn_mat.dtype) == np.bool_
  return ShapedArray((conn_mat.shape[0] // events.shape[0],), value.dtype)


event_add_prim.def_abstract_eval(_event_add_abstract)
event_add_prim.def_impl(partial(xla.apply_primitive, event_add_prim))


def _event_slice_add_abstract(events, post_ids, pre_slice, value, out):
  assert dtypes.canonicalize_dtype(events.dtype) == np.bool_
  return out


event_slice_add_prim.def_abstract_eval(_event_slice_add_abstract)
event_slice_add_prim.def_impl(partial(xla.apply_primitive, event_slice_add_prim))


# We also need a translation rule to convert the function into an XLA op. In
# our case this is the custom XLA op that we've written. We're wrapping two
# translation rules into one here: one for the CPU and one for the GPU
def _event_add_translation(c, events, conn_mat, value, *, platform="cpu"):
  # Extract the dtype and shape
  events_shape = c.get_shape(events)
  events_dim = events_shape.dimensions()
  conn_mat_shape = c.get_shape(conn_mat)
  conn_mat_dim = conn_mat_shape.dimensions()
  value_shape = c.get_shape(value)
  dtype = value_shape.element_type()

  # The event shape
  _events_shape = xla_client.Shape.array_shape(
    events_shape.element_type(), events_dim, (0,))

  # The pre_size shape
  pre_size = np.array(events_dim[0], dtype=np.int64)
  pre_shape = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())

  # The post_size shape
  post_size = np.array(conn_mat_dim[0] // events_dim[0], dtype=np.int64)
  post_shape = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())

  # The conn_mat shape
  _conn_mat_shape = xla_client.Shape.array_shape(
    conn_mat_shape.element_type(), conn_mat_dim, (0,))

  # The value shape
  _value_shape = xla_client.Shape.array_shape(
    dtype, (), ())

  # The output value shape
  _out_shape = xla_client.Shape.array_shape(
    dtype, (post_size,), (0,))

  # We dispatch a different call depending on the dtype
  if dtype == np.float32:
    op_name = platform.encode() + b"_event_add_f32"
  elif dtype == np.float64:
    op_name = platform.encode() + b"_event_add_f64"
  else:
    raise NotImplementedError(f"Unsupported dtype {dtype}")

  # And then the following is what changes between the GPU and CPU
  if platform == "cpu":
    # On the CPU, we pass the size of the data as a the first input
    # argument
    return xla_client.ops.CustomCallWithLayout(
      c,  # builder
      op_name,  # call_target_name
      # The inputs:
      operands=(xla_client.ops.ConstantLiteral(c, pre_size),
                xla_client.ops.ConstantLiteral(c, post_size),
                events,
                conn_mat,
                value),
      # The input shapes:
      operand_shapes_with_layout=(pre_shape,
                                  post_shape,
                                  _events_shape,
                                  _conn_mat_shape,
                                  _value_shape),
      # The output shapes:
      shape_with_layout=_out_shape,
    )
  raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


# Connect the XLA translation rules for JIT compilation
xla.backend_specific_translations["cpu"][event_add_prim] = \
  partial(_event_add_translation, platform="cpu")


def _event_slice_add_translation(c, events, post_ids, pre_slice, value, out, *, platform="cpu"):
  # Extract the dtype and shape

  # The event shape
  events_shape = c.get_shape(events)
  events_dim = events_shape.dimensions()
  _events_shape = xla_client.Shape.array_shape(
    events_shape.element_type(), events_dim, (0,))

  # The pre_size shape
  pre_size = np.array(events_dim[0], dtype=np.int64)
  _pre_shape = xla_client.Shape.array_shape(np.dtype(np.int64), (), ())

  # The post_ids shape
  post_ids_shape = c.get_shape(post_ids)
  _post_ids_shape = xla_client.Shape.array_shape(
    post_ids_shape.element_type(), post_ids_shape.dimensions(), (0,))

  # The post_ids shape
  pre_slice_shape = c.get_shape(pre_slice)
  _pre_slice_shape = xla_client.Shape.array_shape(
    pre_slice_shape.element_type(), pre_slice_shape.dimensions(), (0,))

  # The value shape
  value_shape = c.get_shape(value)
  dtype = value_shape.element_type()
  _value_shape = xla_client.Shape.array_shape(dtype, (), ())

  # The output value shape
  out_shape = c.get_shape(out)
  _out_shape = xla_client.Shape.array_shape(
    dtype, out_shape.dimensions(), (0,))

  # We dispatch a different call depending on the dtype
  if dtype == np.float32:
    op_name = platform.encode() + b"_event_slice_add_f32"
  elif dtype == np.float64:
    op_name = platform.encode() + b"_event_slice_add_f64"
  else:
    raise NotImplementedError(f"Unsupported dtype {dtype}")

  # And then the following is what changes between the GPU and CPU
  if platform == "cpu":
    # On the CPU, we pass the size of the data as a the first input
    # argument
    return xla_client.ops.CustomCallWithLayout(
      c,  # builder
      op_name,  # call_target_name
      operands=(xla_client.ops.ConstantLiteral(c, pre_size),
                events,
                post_ids,
                pre_slice,
                value),  # The inputs
      operand_shapes_with_layout=(_pre_shape,
                                  _events_shape,
                                  _post_ids_shape,
                                  _pre_slice_shape,
                                  _value_shape),  # The input shapes
      shape_with_layout=_out_shape,  # The output shapes
    )
  raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


# Connect the XLA translation rules for JIT compilation
xla.backend_specific_translations["cpu"][event_slice_add_prim] = \
  partial(_event_slice_add_translation, platform="cpu")
