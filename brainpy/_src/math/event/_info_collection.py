# -*- coding: utf-8 -*-

from typing import Tuple, Union

import jax
import numba
from jax import dtypes, numpy as jnp
from jax.core import ShapedArray
from jax.lib import xla_client

from brainpy._src.dependency_check import import_brainpylib_gpu_ops
from brainpy._src.dependency_check import import_taichi
from brainpy._src.math.interoperability import as_jax
from brainpy._src.math.ndarray import Array
from brainpy._src.math.op_register.base import XLACustomOp
from brainpy.errors import GPUOperatorNotFound

ti = import_taichi()

__all__ = [
  'info'
]


def info(events: Union[Array, jax.Array]) -> Tuple[jax.Array, jax.Array]:
  """Collect event information, including event indices, and event number.

  This function supports JAX transformations, including `jit()`,
  `vmap()` and `pmap()`.

  Parameters
  ----------
  events: jax.Array
    The events.

  Returns
  -------
  res: tuple
    A tuple with two elements, denoting the event indices and the event number.
  """
  events = as_jax(events)
  if events.ndim != 1:
    raise TypeError('Only support 1D boolean vector.')
  return event_info_p(events)


def _batch_event_info_abstract(events):
  assert events.ndim == 2
  # assert events.dtype == jnp.bool_
  event_ids = ShapedArray(dtype=dtypes.canonicalize_dtype(int), shape=events.shape)
  event_num = ShapedArray(dtype=dtypes.canonicalize_dtype(int), shape=(events.shape[0],))
  return event_ids, event_num


@numba.njit(fastmath=True, parallel=True, nogil=True)
def _batch_event_info(outs, ins):
  event_ids, event_num = outs
  event_num.fill(0)
  event_ids.fill(-1)
  events = ins
  for batch_idx in range(event_ids.shape[0]):
    num = 0
    for i in range(event_ids.shape[1]):
      if events[batch_idx, i]:
        event_ids[batch_idx, num] = i
        num += 1
    event_num[batch_idx] = num


@ti.kernel
def _batch_event_info_taichi(events: ti.types.ndarray(ndim=2),
                             event_ids: ti.types.ndarray(ndim=2),
                             event_num: ti.types.ndarray(ndim=1)):
  for i, j in ti.grouped(ti.ndrange(event_ids.shape)):
    event_ids[i, j] = -1
  for batch_idx in range(event_ids.shape[0]):
    num = 0
    for i in range(event_ids.shape[1]):
      if events[batch_idx, i]:
        event_ids[batch_idx, num] = i
        num += 1
    event_num[batch_idx] = num


def _batch_event_info_batching_rule(args, axes):
  arg = jnp.moveaxis(args[0], axes[0], 0)
  shape = arg.shape
  arg = jnp.reshape(arg, (shape[0] * shape[1], shape[2]))
  event_ids, event_num = batch_event_info_p(arg)
  return ((jnp.reshape(event_ids, shape), jnp.reshape(event_num, shape[:2])),
          (0, 0))


def _event_info_gpu_translation(c, events):
  gpu_ops = import_brainpylib_gpu_ops()
  if gpu_ops is None:
    raise GPUOperatorNotFound(event_info_p.name)

  e_shape = c.get_shape(events).dimensions()
  e_type = c.get_shape(events).element_type()
  if len(e_shape) == 1:
    event_size = e_shape[0]
    batch_size = 1
    event_ids_shape = xla_client.Shape.array_shape(dtypes.canonicalize_dtype(int),
                                                   (event_size,),
                                                   (0,))
  else:
    batch_size, event_size = e_shape
    event_ids_shape = xla_client.Shape.array_shape(dtypes.canonicalize_dtype(int),
                                                   (batch_size, event_size),
                                                   (1, 0))
  event_num_shape = xla_client.Shape.array_shape(dtypes.canonicalize_dtype(int),
                                                 (batch_size,),
                                                 (0,))
  opaque = gpu_ops.build_nonzero_descriptor(event_size, batch_size)

  if e_type == jnp.bool_:
    type_name = b'_bool'
  elif e_type == jnp.int32:
    type_name = b'_int'
  elif e_type == jnp.int64:
    type_name = b'_long'
  elif e_type == jnp.float32:
    type_name = b'_float'
  elif e_type == jnp.float64:
    type_name = b'_double'
  else:
    raise ValueError

  return xla_client.ops.CustomCallWithLayout(
    c,
    b'nonzero' + type_name,
    operands=(events,),
    operand_shapes_with_layout=(c.get_shape(events),),
    shape_with_layout=xla_client.Shape.tuple_shape((event_ids_shape, event_num_shape)),
    opaque=opaque,
  )


batch_event_info_p = XLACustomOp(
  name='batched_event_info',
  cpu_kernel=_batch_event_info_taichi,
  outs=_batch_event_info_abstract,
)
batch_event_info_p.def_batching_rule(_batch_event_info_batching_rule)


def _event_info_abstract(events, **kwargs):
  assert events.ndim == 1
  # assert events.dtype == jnp.bool_
  event_ids = ShapedArray(dtype=dtypes.canonicalize_dtype(int), shape=events.shape)
  event_num = ShapedArray(dtype=dtypes.canonicalize_dtype(int), shape=(1,))
  return event_ids, event_num


# TODO: first parallel evaluate the sub-sections, then serially event the sub-results.
@numba.jit(fastmath=True)
def _event_info(outs, ins):
  event_ids, event_num = outs
  event_num.fill(0)
  event_ids.fill(-1)
  events = ins
  num = 0
  for i in range(event_ids.shape[0]):
    if events[i]:
      event_ids[num] = i
      num += 1
  event_num[0] = num


@ti.kernel
def _event_info_taichi(events: ti.types.ndarray(ndim=1),
                       event_ids: ti.types.ndarray(ndim=1),
                       event_num: ti.types.ndarray(ndim=1)):
  for i in range(event_ids.shape[0]):
    event_ids[i] = -1
  num = 0
  for i in range(event_ids.shape[0]):
    if events[i]:
      event_ids[num] = i
      num += 1
  event_num[0] = num


def _event_info_batching_rule(args, axes):
  arg = jnp.moveaxis(args[0], axes[0], 0)
  return (batch_event_info_p(arg), (0, 0))


event_info_p = XLACustomOp(
  name='event_info',
  cpu_kernel=_event_info_taichi,
  outs=_event_info_abstract,
  # gpu_func_translation=_event_info_gpu_translation,
)
event_info_p.def_batching_rule(_event_info_batching_rule)
