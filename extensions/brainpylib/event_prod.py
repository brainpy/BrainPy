# -*- coding: utf-8 -*-

__all__ = [
  'event_prod',
]

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import core
from jax.interpreters import xla
from jax.lib import xla_client

try:
  from . import gpu_ops
except ImportError:
  gpu_ops = None

x_shape = xla_client.Shape.array_shape
x_ops = xla_client.ops

_event_prod_prim = core.Primitive("event_prod")


def event_prod(events, pre2post, post_num, values):
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
  values = jnp.asarray([values])
  if values.dtype not in [jnp.float32, jnp.float64]:
    raise ValueError(f'The dtype of "values" must be float32 or float64, while we got {values.dtype}.')
  if values.size not in [1, indices.size]:
    raise ValueError(f'The size of "values" must be 1 (a scalar) or len(pre2post[0]) (a vector), '
                     f'while we got {values.size} != 1 != {indices.size}')
  out = jnp.zeros(post_num, dtype=values.dtype)
  values = values.flatten()
  # bind operator
  return _event_prod_prim.bind(events, indices, indptr, values, out)


def _event_prod_abstract(events, indices, indptr, values, out):
  return out


_event_prod_prim.def_abstract_eval(_event_prod_abstract)
_event_prod_prim.def_impl(partial(xla.apply_primitive, _event_prod_prim))


def _event_prod_translation(c, events, indices, indptr, values, out, *, platform="cpu"):
  # The pre/post shape
  pre_size = np.array(c.get_shape(events).dimensions()[0], dtype=np.uint32)
  post_size = np.array(c.get_shape(out).dimensions()[0], dtype=np.uint32)
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

  # We dispatch a different call depending on the dtype
  f_type = b'_f32' if Ftype == np.float32 else b'_f64'
  i_type = b'_i32' if Itype == np.uint32 else b'_i64'

  # And then the following is what changes between the GPU and CPU
  if platform == "cpu":
    v_type = b'_event_prod_homo' if values_dim[0] == 1 else b'_event_prod_heter'
    return x_ops.CustomCallWithLayout(
      c, platform.encode() + v_type + f_type + i_type,
      operands=(x_ops.ConstantLiteral(c, pre_size),
                x_ops.ConstantLiteral(c, post_size),
                events, indices, indptr, values),
      operand_shapes_with_layout=(_pre_shape, _post_shape, c.get_shape(events),
                                  c.get_shape(indices), c.get_shape(indptr),
                                  c.get_shape(values)),
      shape_with_layout=c.get_shape(out),
    )
  elif platform == 'gpu':
    if gpu_ops is None:
      raise ValueError('Cannot find compiled gpu wheels.')
    v_type = b'_event_prod_homo' if values_dim[0] == 1 else b'_event_prod_heter'
    opaque = gpu_ops.build_event_prod_descriptor(pre_size, post_size)
    return x_ops.CustomCallWithLayout(
      c, platform.encode() + v_type + f_type + i_type,
      operands=(events, indices, indptr, values),
      operand_shapes_with_layout=(c.get_shape(events), c.get_shape(indices),
                                  c.get_shape(indptr), c.get_shape(values)),
      shape_with_layout=c.get_shape(out),
      opaque=opaque,
    )

  else:
    raise ValueError("Unsupported platform, we only support 'cpu' or 'gpu'")


xla.backend_specific_translations["cpu"][_event_prod_prim] = partial(_event_prod_translation, platform="cpu")
xla.backend_specific_translations["gpu"][_event_prod_prim] = partial(_event_prod_translation, platform="gpu")


