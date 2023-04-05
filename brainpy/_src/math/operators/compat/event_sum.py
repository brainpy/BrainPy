# -*- coding: utf-8 -*-

__all__ = [
  'csr_event_sum', 'coo_event_sum',
]

from functools import partial
from typing import Union, Tuple

import jax.numpy as jnp
import numpy as np
from jax import core
from jax.interpreters import xla
from jax.lib import xla_client

from brainpy._src.math.operators.errors import GPUOperatorNotFound
from brainpy._src.math.operators.op_registers import utils
from brainpy._src.math.operators.event_ops.event_csr_matvec import event_csr_matvec
from brainpy._src.math.operators.tools import transform_brainpy_array

try:
  from brainpy import gpu_ops
except ImportError:
  gpu_ops = None

x_shape = xla_client.Shape.array_shape
x_ops = xla_client.ops

csr_event_sum_p1 = core.Primitive("csr_event_sum_p1")


def csr_event_sum(events: jnp.ndarray,
                  pre2post: Tuple[jnp.ndarray, jnp.ndarray],
                  post_num: int,
                  values: Union[float, jnp.ndarray]):
  events = transform_brainpy_array(events)
  post_num = transform_brainpy_array(post_num)
  values = transform_brainpy_array(values)
  indices, indptr = pre2post
  indices = transform_brainpy_array(indices)
  indptr = transform_brainpy_array(indptr)
  # events
  if events.dtype != jnp.bool_:
    raise ValueError(f'"events" must be a vector of bool, while we got {events.dtype}')

  # connections
  if len(events) + 1 != len(indptr):
    raise ValueError(f'The length of "events" must be equal to "len(pre2post[1]) - 1", '
                     f'while we get: {len(events)} + 1 != {len(indptr)}')
  if indices.dtype != indptr.dtype:
    raise ValueError(f"The dtype of pre2post[0] must be equal to that of pre2post[1], "
                     f"while we got {(indices.dtype, indptr.dtype)}")
  if not jnp.issubdtype(indices.dtype, jnp.integer):
    raise ValueError(f'The dtype of pre2post must be integer, while we got {indices.dtype}')

  # output value
  if np.ndim(values) == 0:
    values = jnp.asarray([values])
  dtype = values.dtype
  if dtype not in [jnp.float32, jnp.float64]:
    raise ValueError(f'The dtype of "values" must be float32 or float64, while we got {dtype}.')
  if np.size(values) not in [1, indices.size]:
    raise ValueError(f'The size of "values" must be 1 (a scalar) or len(pre2post[0]) (a vector), '
                     f'while we got {np.size(values)} != 1 != {indices.size}')
  # bind operator
  return event_csr_matvec(values, indices, indptr, events, shape=(events.shape[0], post_num), transpose=True)


def _event_sum_abstract(events, indices, indptr, values, *, post_num):
  return core.ShapedArray(dtype=values.dtype, shape=(post_num,))


def _event_sum_translation(c, events, indices, indptr, values, *, post_num, platform="cpu"):
  # The shape of pre/post
  pre_size = np.array(c.get_shape(events).dimensions()[0], dtype=np.uint32)
  _pre_shape = x_shape(np.dtype(np.uint32), (), ())
  _post_shape = x_shape(np.dtype(np.uint32), (), ())

  # The indices shape
  indices_shape = c.get_shape(indices)
  Itype = indices_shape.element_type()

  # The value shape
  values_shape = c.get_shape(values)
  Ftype = values_shape.element_type()
  values_dim = values_shape.dimensions()

  # We dispatch a different call depending on the dtype
  f_type = b'_f32' if Ftype == np.float32 else b'_f64'
  i_type = b'_i32' if Itype in [np.uint32, np.int32] else b'_i64'

  if platform == "cpu":
    v_type = b'cpu_csr_event_sum_homo' if values_dim[0] == 1 else b'cpu_csr_event_sum_heter'
    return x_ops.CustomCallWithLayout(
      c,
      v_type + f_type + i_type,
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

  # GPU platform
  elif platform == 'gpu':
    if gpu_ops is None:
      raise GPUOperatorNotFound('event_sum')

    v_type = b'gpu_csr_event_sum_homo' if values_dim[0] == 1 else b'gpu_csr_event_sum_heter'
    opaque = gpu_ops.build_csr_event_sum_descriptor(pre_size, post_num)
    return x_ops.CustomCallWithLayout(
      c,
      v_type + f_type + i_type,
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


utils.register_general_batching(csr_event_sum_p1)
csr_event_sum_p1.def_abstract_eval(_event_sum_abstract)
csr_event_sum_p1.def_impl(partial(xla.apply_primitive, csr_event_sum_p1))
xla.backend_specific_translations["cpu"][csr_event_sum_p1] = partial(_event_sum_translation, platform="cpu")
xla.backend_specific_translations["gpu"][csr_event_sum_p1] = partial(_event_sum_translation, platform="gpu")

# ---------------------------
# event sum kernel 2
# ---------------------------

coo_event_sum_p1 = core.Primitive("coo_event_sum_p1")


def coo_event_sum(events, pre_ids, post_ids, post_num, values):
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
  if not jnp.issubdtype(pre_ids.dtype, jnp.integer):
    raise ValueError(f'The dtype of "post_ids/pre_ids" must be a subtype of integer, '
                     f'while we got {pre_ids.dtype}')

  # output value
  if np.ndim(values) == 0:
    values = jnp.asarray([values])
  if values.dtype not in [jnp.float32, jnp.float64]:
    raise ValueError(f'The dtype of "values" must be float32 or float64, while we got {values.dtype}.')
  if values.size not in [1, pre_ids.size]:
    raise ValueError(f'The size of "values" must be 1 (a scalar) or len(pre_ids) (a vector), '
                     f'while we got {values.size} != 1 != {pre_ids.size}')
  values = values.flatten()

  # bind operator
  return coo_event_sum_p1.bind(events, pre_ids, post_ids, values, post_num=post_num)


def _event_sum2_abstract(events, pre_ids, post_ids, value, *, post_num):
  return core.ShapedArray(dtype=value.dtype, shape=(post_num,))


def _event_sum2_translation(c, events, pre_ids, post_ids, values, *, post_num, platform="cpu"):
  # The conn/post shape
  conn_size = np.array(c.get_shape(pre_ids).dimensions()[0], dtype=np.uint32)
  _pre_shape = x_shape(np.dtype(np.uint32), (), ())
  _post_shape = x_shape(np.dtype(np.uint32), (), ())

  # The pre_ids shape
  pre_ids_shape = c.get_shape(pre_ids)
  Itype = pre_ids_shape.element_type()
  assert np.issubdtype(Itype, np.integer)

  # The value shape
  values_shape = c.get_shape(values)
  Ftype = values_shape.element_type()
  assert Ftype in [np.float32, np.float64]
  values_dim = values_shape.dimensions()

  # We dispatch a different call depending on the dtype
  f_type = b'_f32' if Ftype == np.float32 else b'_f64'
  i_type = b'_i32' if Itype in [np.uint32, np.int32, jnp.uint32, jnp.int32] else b'_i64'

  # And then the following is what changes between the GPU and CPU
  if platform == "cpu":
    v_type = b'cpu_coo_event_sum_homo' if values_dim[0] == 1 else b'cpu_coo_event_sum_heter'
    return x_ops.CustomCallWithLayout(
      c,
      v_type + f_type + i_type,
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
      raise GPUOperatorNotFound(coo_event_sum_p1.name)
    v_type = b'gpu_coo_event_sum_homo' if values_dim[0] == 1 else b'gpu_coo_event_sum_heter'
    opaque = gpu_ops.build_csr_event_sum_descriptor(conn_size, post_num)
    return x_ops.CustomCallWithLayout(
      c,
      v_type + f_type + i_type,
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


coo_event_sum_p1.def_abstract_eval(_event_sum2_abstract)
coo_event_sum_p1.def_impl(partial(xla.apply_primitive, coo_event_sum_p1))
xla.backend_specific_translations["cpu"][coo_event_sum_p1] = partial(_event_sum2_translation, platform="cpu")
xla.backend_specific_translations["gpu"][coo_event_sum_p1] = partial(_event_sum2_translation, platform="gpu")
utils.register_general_batching(coo_event_sum_p1)
