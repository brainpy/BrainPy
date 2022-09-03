# -*- coding: utf-8 -*-

__all__ = [
  'event_sum',
  'event_sum2',
]

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import core
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla, batching
from jax.lax import scan
from jax.lib import xla_client

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
  values = jnp.asarray([values])
  if values.dtype not in [jnp.float32, jnp.float64]:
    raise ValueError(f'The dtype of "values" must be float32 or float64, while we got {values.dtype}.')
  if values.size not in [1, indices.size]:
    raise ValueError(f'The size of "values" must be 1 (a scalar) or len(pre2post[0]) (a vector), '
                     f'while we got {values.size} != 1 != {indices.size}')
  values = values.flatten()
  # bind operator
  return _event_sum_prim.bind(events, indices, indptr, values, post_num=post_num)


def _event_sum_abstract(events, indices, indptr, values, *, post_num):
  return ShapedArray(dtype=values.dtype, shape=(post_num,))


_event_sum_prim.def_abstract_eval(_event_sum_abstract)
_event_sum_prim.def_impl(partial(xla.apply_primitive, _event_sum_prim))


def _event_sum_translation(c, events, indices, indptr, values, *, post_num, platform="cpu"):
  # The pre/post shape
  pre_size = np.array(c.get_shape(events).dimensions()[0], dtype=np.uint32)
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
    v_type = b'_event_sum_homo' if values_dim[0] == 1 else b'_event_sum_heter'
    return x_ops.CustomCallWithLayout(
      c,
      platform.encode() + v_type + f_type + i_type,
      operands=(x_ops.ConstantLiteral(c, pre_size),
                x_ops.ConstantLiteral(c, post_num),
                events,
                indices,
                indptr,
                values),
      operand_shapes_with_layout=(_pre_shape,
                                  _post_shape,
                                  c.get_shape(events),
                                  c.get_shape(indices),
                                  c.get_shape(indptr),
                                  c.get_shape(values)),
      shape_with_layout=x_shape(np.dtype(Ftype), (post_num,), (0,)),
    )
  elif platform == 'gpu':
    if gpu_ops is None:
      raise ValueError('Cannot find compiled gpu wheels.')
    v_type = b'_event_sum_homo' if values_dim[0] == 1 else b'_event_sum_heter'
    opaque = gpu_ops.build_event_sum_descriptor(pre_size, post_num)
    return x_ops.CustomCallWithLayout(
      c,
      platform.encode() + v_type + f_type + i_type,
      operands=(events,
                indices,
                indptr,
                values),
      operand_shapes_with_layout=(c.get_shape(events),
                                  c.get_shape(indices),
                                  c.get_shape(indptr),
                                  c.get_shape(values)),
      shape_with_layout=x_shape(np.dtype(Ftype), (post_num,), (0,)),
      opaque=opaque,
    )

  else:
    raise ValueError("Unsupported platform, we only support 'cpu' or 'gpu'")


xla.backend_specific_translations["cpu"][_event_sum_prim] = partial(_event_sum_translation, platform="cpu")
xla.backend_specific_translations["gpu"][_event_sum_prim] = partial(_event_sum_translation, platform="gpu")


def _event_sum_batch(args, axes):
  batch_axes, batch_args, non_batch_args = [], {}, {}
  for ax_i, ax in enumerate(axes):
    if ax is None:
      non_batch_args[f'ax{ax_i}'] = args[ax_i]
    else:
      batch_args[f'ax{ax_i}'] = args[ax_i] if ax == 0 else jnp.moveaxis(args[ax_i], ax, 0)
      batch_axes.append(ax_i)

  def f(_, x):
    pars = tuple([(x[f'ax{i}'] if i in batch_axes else non_batch_args[f'ax{i}'])
                  for i in range(len(axes))])
    return 0, _event_sum_prim.bind(*pars)
  _, outs = scan(f, 0, batch_args)
  return outs, 0


batching.primitive_batchers[_event_sum_prim] = _event_sum_batch


# ---------------------------
# event sum kernel 2
# ---------------------------


_event_sum2_prim = core.Primitive("event_sum2")


def event_sum2(events, pre_ids, post_ids, post_num, values):
  # events
  if events.dtype != jnp.bool_:
    raise ValueError(f'"events" must be a vector of bool, while we got {events.dtype}')

  # connections
  if len(pre_ids) != len(post_ids):
    raise ValueError(f'The length of "pre_ids" must be equal to "post_ids", '
                     f'while we get: {len(pre_ids)} != {len(post_ids)}')
  if pre_ids.dtype != post_ids.dtype:
    raise ValueError(f'The dtype of "pre_ids" must be equal to that of "post_ids", '
                     f'while we got {(pre_ids.dtype, post_ids.dtype)}')
  if pre_ids.dtype not in [jnp.uint32, jnp.uint64]:
    raise ValueError(f'The dtype of "post_ids/pre_ids" must be uint32 or uint64, '
                     f'while we got {pre_ids.dtype}')

  # output value
  values = jnp.asarray([values])
  if values.dtype not in [jnp.float32, jnp.float64]:
    raise ValueError(f'The dtype of "values" must be float32 or float64, while we got {values.dtype}.')
  if values.size not in [1, pre_ids.size]:
    raise ValueError(f'The size of "values" must be 1 (a scalar) or len(pre_ids) (a vector), '
                     f'while we got {values.size} != 1 != {pre_ids.size}')
  values = values.flatten()

  # bind operator
  return _event_sum2_prim.bind(events, pre_ids, post_ids, values, post_num=post_num)


def _event_sum2_abstract(events, pre_ids, post_ids, value, *, post_num):
  return ShapedArray(dtype=value.dtype, shape=(post_num,))


_event_sum2_prim.def_abstract_eval(_event_sum2_abstract)
_event_sum2_prim.def_impl(partial(xla.apply_primitive, _event_sum2_prim))


def _event_sum2_translation(c, events, pre_ids, post_ids, values, *, post_num, platform="cpu"):
  # The conn/post shape
  conn_size = np.array(c.get_shape(pre_ids).dimensions()[0], dtype=np.uint32)
  _pre_shape = x_shape(np.dtype(np.uint32), (), ())
  _post_shape = x_shape(np.dtype(np.uint32), (), ())

  # The pre_ids shape
  pre_ids_shape = c.get_shape(pre_ids)
  Itype = pre_ids_shape.element_type()
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
    v_type = b'_event_sum2_homo' if values_dim[0] == 1 else b'_event_sum2_heter'
    return x_ops.CustomCallWithLayout(
      c,
      platform.encode() + v_type + f_type + i_type,
      operands=(x_ops.ConstantLiteral(c, conn_size),
                x_ops.ConstantLiteral(c, post_num),
                events,
                pre_ids,
                post_ids,
                values),
      operand_shapes_with_layout=(_pre_shape,
                                  _post_shape,
                                  c.get_shape(events),
                                  c.get_shape(pre_ids),
                                  c.get_shape(post_ids),
                                  c.get_shape(values)),
      shape_with_layout=x_shape(np.dtype(Ftype), (post_num,), (0,)),
    )
  elif platform == 'gpu':
    if gpu_ops is None:
      raise ValueError('Cannot find compiled gpu wheels.')
    v_type = b'_event_sum2_homo' if values_dim[0] == 1 else b'_event_sum2_heter'
    opaque = gpu_ops.build_event_sum2_descriptor(conn_size, post_num)
    return x_ops.CustomCallWithLayout(
      c,
      platform.encode() + v_type + f_type + i_type,
      operands=(events,
                pre_ids,
                post_ids,
                values),
      operand_shapes_with_layout=(c.get_shape(events),
                                  c.get_shape(pre_ids),
                                  c.get_shape(post_ids),
                                  c.get_shape(values)),
      shape_with_layout=x_shape(np.dtype(Ftype), (post_num,), (0,)),
      opaque=opaque,
    )
  raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


xla.backend_specific_translations["cpu"][_event_sum2_prim] = partial(_event_sum2_translation, platform="cpu")
xla.backend_specific_translations["gpu"][_event_sum2_prim] = partial(_event_sum2_translation, platform="gpu")
