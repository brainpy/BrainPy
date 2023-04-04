# -*- coding: utf-8 -*-

__all__ = [
  'coo_atomic_sum',
]

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import core
from jax.interpreters import xla
from jax.lib import xla_client
from brainpy._src.math.operators.errors import GPUOperatorNotFound
from brainpy._src.math.operators.tools import transform_brainpy_array


try:
  from brainpy import gpu_ops
except ImportError:
  gpu_ops = None


coo_atomic_sum_p1 = core.Primitive("coo_atomic_sum_p1")


def coo_atomic_sum(values, post_ids, post_num, pre_ids=None):
  values = transform_brainpy_array(values)
  post_ids = transform_brainpy_array(post_ids)
  post_num = transform_brainpy_array(post_num)
  pre_ids = transform_brainpy_array(pre_ids)
  # connections
  if jnp.size(values) != 1:
    assert pre_ids is not None, 'Must provide "pre_ids" when "values" is not a scalar.'
  else:
    pre_ids = post_ids
  if len(pre_ids) != len(post_ids):
    raise ValueError(f'The length of "pre_ids" and "post_ids" must be the same, '
                     f'while we got {len(pre_ids)} != {len(post_ids)}')
  if pre_ids.dtype != post_ids.dtype:
    raise ValueError(f"The dtype of pre_ids must be equal to that of post_ids, "
                     f"while we got {(pre_ids.dtype, post_ids.dtype)}")
  if not jnp.issubdtype(post_ids.dtype, jnp.integer):
    raise ValueError(f'The dtype of post_ids must be a subtype of integer, while we got {post_ids.dtype}')

  # output value
  if np.ndim(values) == 0:
    values = jnp.asarray([values])
  if values.dtype not in [jnp.float32, jnp.float64]:
    raise ValueError(f'The dtype of "values" must be float32 or float64, while we got {values.dtype}.')
  # if values.size not in [1, pre_ids.size]:
  #   raise ValueError(f'The size of "values" must be 1 (a scalar) or len(pre_ids) (a vector), '
  #                    f'while we got {values.size} != 1 != {pre_ids.size}')
  if values.size != 1 and values.size <= pre_ids.max():
    raise ValueError(f'The size of "values" must be 1 (a scalar) or longer than pre_size (a vector), '
                     f'while we got {values.size} != 1 <= {pre_ids.max()}')

  # bind operator
  return coo_atomic_sum_p1.bind(values, pre_ids, post_ids, post_num=post_num)


def _atomic_sum_abstract(values, pre_ids, post_ids, *, post_num):
  return core.ShapedArray(dtype=values.dtype, shape=(post_num,))


coo_atomic_sum_p1.def_abstract_eval(_atomic_sum_abstract)
coo_atomic_sum_p1.def_impl(partial(xla.apply_primitive, coo_atomic_sum_p1))


def _atomic_sum_translation(c, values, pre_ids, post_ids, *, post_num, platform="cpu"):
  # The conn/post shape
  conn_size = np.array(c.get_shape(post_ids).dimensions()[0], dtype=np.uint32)
  _conn_shape = xla_client.Shape.array_shape(np.dtype(np.uint32), (), ())
  _out_shape = xla_client.Shape.array_shape(np.dtype(np.uint32), (), ())

  # The indices shape
  Itype = c.get_shape(post_ids).element_type()
  assert np.issubdtype(Itype, np.integer)

  # The value shape
  values_info = c.get_shape(values)
  values_dtype = values_info.element_type()
  assert values_dtype in [np.float32, np.float64]

  # We dispatch a different call depending on the dtype
  values_dim = values_info.dimensions()
  v_type = b'_coo_atomic_sum_homo' if (values_dim[0] == 1) else b'_coo_atomic_sum_heter'
  f_type = b'_f32' if values_dtype == np.float32 else b'_f64'
  i_type = b'_i32' if Itype in [np.uint32, np.int32, jnp.uint32, jnp.int32] else b'_i64'

  # And then the following is what changes between the GPU and CPU
  if platform == "cpu":
    if values_dim[0] != 1:
      return xla_client.ops.CustomCallWithLayout(
        c, platform.encode() + v_type + f_type + i_type,
        operands=(values,
                  pre_ids,
                  post_ids,
                  xla_client.ops.ConstantLiteral(c, conn_size),
                  xla_client.ops.ConstantLiteral(c, post_num)),
        operand_shapes_with_layout=(c.get_shape(values),
                                    c.get_shape(pre_ids),
                                    c.get_shape(post_ids),
                                    _conn_shape,
                                    _out_shape),
        shape_with_layout=xla_client.Shape.array_shape(np.dtype(values_dtype), (post_num,), (0,)),
      )
    else:
      return xla_client.ops.CustomCallWithLayout(
        c, platform.encode() + v_type + f_type + i_type,
        operands=(values,
                  post_ids,
                  xla_client.ops.ConstantLiteral(c, conn_size),
                  xla_client.ops.ConstantLiteral(c, post_num)),
        operand_shapes_with_layout=(c.get_shape(values),
                                    c.get_shape(post_ids),
                                    _conn_shape,
                                    _out_shape),
        shape_with_layout=xla_client.Shape.array_shape(np.dtype(values_dtype), (post_num,), (0,)),
      )
  elif platform == 'gpu':
    if gpu_ops is None:
      raise GPUOperatorNotFound(coo_atomic_sum_p1)

    opaque = gpu_ops.build_coo_atomic_sum_descriptor(conn_size, post_num)
    if values_dim[0] != 1:
      return xla_client.ops.CustomCallWithLayout(
        c, platform.encode() + v_type + f_type + i_type,
        operands=(values,
                  pre_ids,
                  post_ids),
        operand_shapes_with_layout=(c.get_shape(values),
                                    c.get_shape(pre_ids),
                                    c.get_shape(post_ids)),
        shape_with_layout=xla_client.Shape.array_shape(np.dtype(values_dtype), (post_num,), (0,)),
        opaque=opaque,
      )
    else:
      return xla_client.ops.CustomCallWithLayout(
        c, platform.encode() + v_type + f_type + i_type,
        operands=(values, post_ids),
        operand_shapes_with_layout=(c.get_shape(values), c.get_shape(post_ids)),
        shape_with_layout=xla_client.Shape.array_shape(np.dtype(values_dtype), (post_num,), (0,)),
        opaque=opaque,
      )

  else:
    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


xla.backend_specific_translations["cpu"][coo_atomic_sum_p1] = partial(_atomic_sum_translation, platform="cpu")
xla.backend_specific_translations["gpu"][coo_atomic_sum_p1] = partial(_atomic_sum_translation, platform="gpu")
