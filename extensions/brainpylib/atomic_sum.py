# -*- coding: utf-8 -*-

__all__ = [
  'atomic_sum',
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

_atomic_sum_prim = core.Primitive("atomic_sum")


def atomic_sum(values, post_ids, post_num, pre_ids=None):
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
  if post_ids.dtype not in [jnp.uint32, jnp.uint64]:
    raise ValueError(f'The dtype of post_ids must be uint32 or uint64, while we got {post_ids.dtype}')

  # output value
  values = jnp.asarray([values])
  if values.dtype not in [jnp.float32, jnp.float64]:
    raise ValueError(f'The dtype of "values" must be float32 or float64, while we got {values.dtype}.')
  # if values.size not in [1, pre_ids.size]:
  #   raise ValueError(f'The size of "values" must be 1 (a scalar) or len(pre_ids) (a vector), '
  #                    f'while we got {values.size} != 1 != {pre_ids.size}')
  if values.size != 1 and values.size <= pre_ids.max():
    raise ValueError(f'The size of "values" must be 1 (a scalar) or longer than pre_size (a vector), '
                     f'while we got {values.size} != 1 <= {pre_ids.max()}')
  values = values.flatten()
  out = jnp.zeros(post_num, dtype=values.dtype)

  # bind operator
  return _atomic_sum_prim.bind(values, pre_ids, post_ids, out)


def _atomic_sum_abstract(values, pre_ids, post_ids, out):
  return out


_atomic_sum_prim.def_abstract_eval(_atomic_sum_abstract)
_atomic_sum_prim.def_impl(partial(xla.apply_primitive, _atomic_sum_prim))


def _atomic_sum_translation(c, values, pre_ids, post_ids, out, *, platform="cpu"):
  # The conn/post shape
  conn_size = np.array(c.get_shape(post_ids).dimensions()[0], dtype=np.uint32)
  out_size = np.array(c.get_shape(out).dimensions()[0], dtype=np.uint32)
  _conn_shape = x_shape(np.dtype(np.uint32), (), ())
  _out_shape = x_shape(np.dtype(np.uint32), (), ())

  # The indices shape
  Itype = c.get_shape(post_ids).element_type()
  assert Itype in [np.uint32, np.uint64]

  # The value shape
  Ftype = c.get_shape(out).element_type()
  assert Ftype in [np.float32, np.float64]

  # We dispatch a different call depending on the dtype
  values_dim = c.get_shape(values).dimensions()
  v_type = b'_atomic_sum_homo' if (values_dim[0] == 1) else b'_atomic_sum_heter'
  f_type = b'_f32' if Ftype == np.float32 else b'_f64'
  i_type = b'_i32' if Itype == np.uint32 else b'_i64'

  # And then the following is what changes between the GPU and CPU
  if platform == "cpu":
    if values_dim[0] != 1:
      return x_ops.CustomCallWithLayout(
        c, platform.encode() + v_type + f_type + i_type,
        operands=(values, pre_ids, post_ids,
                  x_ops.ConstantLiteral(c, conn_size),
                  x_ops.ConstantLiteral(c, out_size)),
        operand_shapes_with_layout=(c.get_shape(values), c.get_shape(pre_ids),
                                    c.get_shape(post_ids), _conn_shape, _out_shape),
        shape_with_layout=c.get_shape(out),
      )
    else:
      return x_ops.CustomCallWithLayout(
        c, platform.encode() + v_type + f_type + i_type,
        operands=(values, post_ids,
                  x_ops.ConstantLiteral(c, conn_size),
                  x_ops.ConstantLiteral(c, out_size)),
        operand_shapes_with_layout=(c.get_shape(values),
                                    c.get_shape(post_ids),
                                    _conn_shape, _out_shape),
        shape_with_layout=c.get_shape(out),
      )
  elif platform == 'gpu':
    if gpu_ops is None: raise ValueError('Cannot find compiled gpu wheels.')

    opaque = gpu_ops.build_atomic_sum_descriptor(conn_size, out_size)
    if values_dim[0] != 1:
      return x_ops.CustomCallWithLayout(
        c, platform.encode() + v_type + f_type + i_type,
        operands=(values, pre_ids, post_ids),
        operand_shapes_with_layout=(c.get_shape(values), c.get_shape(pre_ids), c.get_shape(post_ids)),
        shape_with_layout=c.get_shape(out),
        opaque=opaque,
      )
    else:
      return x_ops.CustomCallWithLayout(
        c, platform.encode() + v_type + f_type + i_type,
        operands=(values, post_ids),
        operand_shapes_with_layout=(c.get_shape(values), c.get_shape(post_ids)),
        shape_with_layout=c.get_shape(out),
        opaque=opaque,
      )

  else:
    raise ValueError("Unsupported platform; this must be either 'cpu' or 'gpu'")


xla.backend_specific_translations["cpu"][_atomic_sum_prim] = partial(_atomic_sum_translation, platform="cpu")
xla.backend_specific_translations["gpu"][_atomic_sum_prim] = partial(_atomic_sum_translation, platform="gpu")
