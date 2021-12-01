# -*- coding: utf-8 -*-

__all__ = [
  'event_add',
]

from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import core, dtypes
from jax.interpreters import xla
from jax.lib import xla_client

# Register the CPU XLA custom calls
from . import cpu_ops
for _name, _value in cpu_ops.registrations().items():
  xla_client.register_cpu_custom_call_target(_name, _value)


# This function exposes the primitive to user code and this is the only
# public-facing function in this module

def event_add(events, post_ids, pre2post_slice, post_num, value):
  value = jnp.array(value)
  out = jnp.zeros(post_num, dtype=value.dtype)
  assert len(events) + 1 == len(pre2post_slice)
  return _event_add_prim.bind(events, post_ids, pre2post_slice, value, out)


_event_add_prim = core.Primitive("event_add")


# *********************************
# *  SUPPORT FOR JIT COMPILATION  *
# *********************************

# For JIT compilation we need a function to evaluate the shape and dtype of the
# outputs of our op for some given inputs
def _event_add_abstract(events, post_ids, pre2post_slice, value, out):
  assert dtypes.canonicalize_dtype(post_ids.dtype) in [np.uint32, np.uint64]
  assert dtypes.canonicalize_dtype(pre2post_slice.dtype) in [np.uint32, np.uint64]
  assert dtypes.canonicalize_dtype(events.dtype) == np.bool_
  assert dtypes.canonicalize_dtype(value.dtype) == dtypes.canonicalize_dtype(out.dtype)
  return out


_event_add_prim.def_abstract_eval(_event_add_abstract)
_event_add_prim.def_impl(partial(xla.apply_primitive, _event_add_prim))


# We also need a translation rule to convert the function into an XLA op. In
# our case this is the custom XLA op that we've written. We're wrapping two
# translation rules into one here: one for the CPU and one for the GPU
def _event_slice_add_translation(c, events, post_ids, pre2post_slice, value, out, *, platform="cpu"):
  # The event shape
  events_shape = c.get_shape(events)
  events_dim = events_shape.dimensions()
  _events_shape = xla_client.Shape.array_shape(
    events_shape.element_type(), events_dim, (0,))

  # The pre_size shape
  pre_size = np.array(events_dim[0], dtype=np.uint32)
  _pre_shape = xla_client.Shape.array_shape(np.dtype(np.uint32), (), ())

  # The post_ids shape
  post_ids_shape = c.get_shape(post_ids)
  _post_ids_shape = xla_client.Shape.array_shape(
    post_ids_shape.element_type(), post_ids_shape.dimensions(), (0,))

  # The pre_slice shape
  pre_slice_shape = c.get_shape(pre2post_slice)
  _pre_slice_shape = xla_client.Shape.array_shape(
    pre_slice_shape.element_type(), pre_slice_shape.dimensions(), (0,))

  # The value shape
  value_shape = c.get_shape(value)
  dtype = value_shape.element_type()
  _value_shape = xla_client.Shape.array_shape(dtype, (), (0,))

  # The output value shape
  out_shape = c.get_shape(out)
  _out_shape = xla_client.Shape.array_shape(
    dtype, out_shape.dimensions(), (0,))

  # We dispatch a different call depending on the dtype
  if dtype == np.float32:
    op_name = platform.encode() + b"_event_add_f32"
  elif dtype == np.float64:
    op_name = platform.encode() + b"_event_add_f64"
  else:
    raise NotImplementedError(f"Unsupported dtype {dtype}")

  # And then the following is what changes between the GPU and CPU
  if platform == "cpu":
    # On the CPU, we pass the size of the data as a the first input argument
    return xla_client.ops.CustomCallWithLayout(
      c,  # builder
      op_name,  # call_target_name
      operands=(xla_client.ops.ConstantLiteral(c, pre_size),
                events,
                post_ids,
                pre2post_slice,
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
xla.backend_specific_translations["cpu"][_event_add_prim] = \
  partial(_event_slice_add_translation, platform="cpu")
